#pragma once

// DC0Engine: ties MCTS + neural network together for UCI integration.
//
// Provides a self-contained search interface that UCI.hpp can call:
//   1. init() — load model, set up evaluator
//   2. new_game() — reset tree
//   3. set_position(board) — set root from current board state
//   4. go(params, info_cb) — run MCTS search, return best move
//   5. stop() — signal search to stop
//
// This file requires libtorch and is only compiled when DC0_ENABLED is defined.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>

#include <Board.hpp>
#include <Piece.hpp>
#include <FEN.hpp>

#include <MoveEncoding.hpp>
#include <BoardEncoding.hpp>
#include <Network.hpp>
#include <MCTS.hpp>
#include <SelfPlay.hpp>
#include <Logging.hpp>

#include <torch/torch.h>

namespace dc0 {

// Parameters for a single search invocation (mapped from UCI go command).
struct SearchParams {
    int simulations = 800;       // target simulations (from go nodes N)
    double movetime = 0.0;       // time limit in seconds (0 = no limit)
    bool infinite = false;       // search until stop
    int batch_size = 64;         // MCTS inference batch size
};

// Info callback: called periodically during search so UCI can emit info lines.
// depth = pseudo-depth (log2 of simulations done), nodes = simulations done,
// score_cp = centipawn score from Q-value, best_move = current best move,
// pv = principal variation (best path from root).
struct SearchInfo {
    int depth = 0;
    int seldepth = 0;
    int64_t nodes = 0;
    int64_t nps = 0;
    int score_cp = 0;
    int time_ms = 0;
    move_t best_move = board::nullmove;
    std::vector<move_t> pv;
};

using InfoCallback = std::function<void(const SearchInfo&)>;
// StopCheck: called between batches; return true to stop search.
using StopCheck = std::function<bool()>;

class DC0Engine {
public:
    DC0Engine() = default;

    // Initialize the neural network model. Returns true on success.
    bool init(int n_blocks, int n_filters, const std::string& model_path,
              const std::string& device_str)
    {
        // Create model
        model_ = dc0::DC0Network(n_blocks, n_filters);

        // Determine device
        if (!device_str.empty()) {
            device_ = torch::Device(device_str);
        } else if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
        } else {
            device_ = torch::kCPU;
        }

        // Load weights if provided
        if (!model_path.empty()) {
            model_->to(torch::kCPU);
            model_->load_weights(model_path);
        }

        // Create evaluator
        evaluator_ = std::make_unique<NNEvaluator>(model_, device_);
        initialized_ = true;
        return true;
    }

    bool is_initialized() const { return initialized_; }

    // Reset tree for a new game.
    void new_game() {
        tree_ = std::make_unique<MCTSTree>();
    }

    // Set the root position. If the position follows from the current tree
    // (i.e., we can advance), reuse the subtree.
    void set_position(const Board& board) {
        if (!tree_) {
            tree_ = std::make_unique<MCTSTree>();
        }
        // Always set fresh root for now.
        // Tree reuse across UCI positions would require tracking the move
        // sequence, which is more complex.
        tree_->set_root(board);
    }

    // Set the root position, attempting to reuse the tree by advancing
    // through the given moves from the starting position.
    void set_position_with_moves(const Board& board,
                                 const std::vector<move_t>& moves_played)
    {
        if (!tree_) {
            tree_ = std::make_unique<MCTSTree>();
            tree_->set_root(board);
            return;
        }
        // If we have a tree and the last move matches, try to advance.
        // For simplicity, just set fresh root. Tree reuse can be added later
        // by tracking the previous position.
        tree_->set_root(board);
    }

    // Run MCTS search and return the best move.
    move_t go(const SearchParams& params,
              const InfoCallback& info_cb,
              const StopCheck& stop_check)
    {
        if (!initialized_ || !evaluator_ || !tree_) {
            return board::nullmove;
        }

        tree_->c_puct = 2.5f;
        tree_->add_noise = false;  // no exploration noise for play

        auto batch_eval_fn = evaluator_->get_batched_eval_function();

        auto t_start = std::chrono::steady_clock::now();
        int sims_done = 0;
        int target_sims = params.simulations;

        // Run simulations in batches, checking stop condition between batches
        while (sims_done < target_sims) {
            // Check stop condition
            if (stop_check && stop_check()) break;

            // Check time limit
            if (params.movetime > 0.0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - t_start).count();
                if (elapsed >= params.movetime) break;
            }

            // Run one batch of simulations
            int batch = std::min(params.batch_size, target_sims - sims_done);
            tree_->run_simulations_batched(batch, batch_eval_fn, params.batch_size);
            sims_done += batch;

            // Emit info periodically (every batch)
            if (info_cb) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - t_start).count();
                int time_ms = static_cast<int>(elapsed * 1000.0);

                SearchInfo info;
                info.nodes = sims_done;
                info.nps = (elapsed > 0.001) ? static_cast<int64_t>(sims_done / elapsed) : 0;
                info.time_ms = time_ms;

                // Compute depth as pseudo-depth (log2 of total visits)
                uint32_t visits = tree_->root_visits();
                info.depth = 0;
                while ((1u << info.depth) < visits && info.depth < 99) info.depth++;

                // Best move and score from root Q-value
                info.best_move = tree_->select_move(0.0f);
                auto stats = tree_->get_move_stats();
                info.score_cp = 0;
                info.seldepth = 0;

                // Find best move stats
                for (auto& s : stats) {
                    if (s.move == info.best_move) {
                        // Q is in [-1, 1], convert to centipawns
                        // Using a logistic mapping: cp = 111.714 * tan(1.5620688 * Q)
                        // Simpler: cp = Q * 128 (linear approximation)
                        float q = s.q_value;
                        info.score_cp = static_cast<int>(q * 128.0f);
                        break;
                    }
                }

                // Build PV: walk down the tree picking most-visited edges
                info.pv = get_pv();

                info_cb(info);
            }
        }

        return tree_->select_move(0.0f);
    }

    // Configuration setters
    void set_simulations(int sims) { default_sims_ = sims; }
    void set_batch_size(int bs) { default_batch_size_ = bs; }
    void set_cpuct(float c) { default_cpuct_ = c; }

    int default_simulations() const { return default_sims_; }
    int default_batch_size() const { return default_batch_size_; }

    // Get principal variation by walking most-visited children from root.
    std::vector<move_t> get_pv() const {
        std::vector<move_t> pv;
        if (!tree_ || !tree_->root()) return pv;

        MCTSNode* node = tree_->root();
        int max_depth = 20;  // limit PV length
        while (node && node->is_expanded && !node->is_terminal
               && !node->edges.empty() && max_depth-- > 0) {
            // Pick most visited edge
            uint32_t best_visits = 0;
            MCTSEdge* best_edge = nullptr;
            for (auto& edge : node->edges) {
                if (edge.visit_count > best_visits) {
                    best_visits = edge.visit_count;
                    best_edge = &edge;
                }
            }
            if (!best_edge || best_visits == 0) break;
            pv.push_back(best_edge->move);
            node = best_edge->child.get();
        }
        return pv;
    }

private:
    DC0Network model_{nullptr};
    torch::Device device_{torch::kCPU};
    std::unique_ptr<NNEvaluator> evaluator_;
    std::unique_ptr<MCTSTree> tree_;
    bool initialized_ = false;

    // Defaults (set via UCI options)
    int default_sims_ = 800;
    int default_batch_size_ = 64;
    float default_cpuct_ = 2.5f;
};

} // namespace dc0
