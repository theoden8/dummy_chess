#pragma once

// dc0 self-play pipeline: generate training data via MCTS + neural network.
//
// Components:
//   - TrainingExample: a single (board_encoding, mcts_policy, game_result) tuple
//   - TrainingData: collection of examples with binary serialization
//   - NNEvaluator: bridges DC0Network to the EvalFunction interface
//   - SelfPlayGame: plays a single game, collecting training examples
//   - SelfPlayManager: manages multiple games, writes output

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <chrono>

#include <m42.h>
#include <Zobrist.hpp>
#include <Board.hpp>
#include <FEN.hpp>

#include <MoveEncoding.hpp>
#include <BoardEncoding.hpp>
#include <MCTS.hpp>
#include <Network.hpp>
#include <Logging.hpp>

#include <torch/torch.h>

namespace dc0 {

// --- Training data ---

struct TrainingExample {
    float planes[ENCODING_SIZE];     // 22 * 64 = 1408 floats
    float policy[POLICY_SIZE];       // 4672 floats
    float result;                    // game result from side-to-move's perspective: {-1, 0, 1}
};

// Binary file format:
//   Header (16 bytes):
//     uint32_t magic = 0xDC000001
//     uint32_t version = 1
//     uint32_t num_examples
//     uint32_t example_size_bytes = sizeof(TrainingExample)
//   Body:
//     TrainingExample[num_examples]
//
// All values are little-endian (native on x86).

static constexpr uint32_t TRAINING_DATA_MAGIC = 0xDC000001;
static constexpr uint32_t TRAINING_DATA_VERSION = 1;

struct TrainingDataHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_examples;
    uint32_t example_size_bytes;
};

class TrainingData {
public:
    std::vector<TrainingExample> examples;

    void add(const TrainingExample& ex) {
        examples.push_back(ex);
    }

    void reserve(size_t n) {
        examples.reserve(n);
    }

    size_t size() const { return examples.size(); }

    // Write all examples to a binary file. Returns true on success.
    bool save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;

        TrainingDataHeader header;
        header.magic = TRAINING_DATA_MAGIC;
        header.version = TRAINING_DATA_VERSION;
        header.num_examples = static_cast<uint32_t>(examples.size());
        header.example_size_bytes = static_cast<uint32_t>(sizeof(TrainingExample));

        bool ok = fwrite(&header, sizeof(header), 1, f) == 1
               && fwrite(examples.data(), sizeof(TrainingExample), examples.size(), f) == examples.size();

        fclose(f);
        return ok;
    }

    // Load examples from a binary file. Returns true on success.
    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;

        TrainingDataHeader header;
        if (fread(&header, sizeof(header), 1, f) != 1) { fclose(f); return false; }

        if (header.magic != TRAINING_DATA_MAGIC
            || header.version != TRAINING_DATA_VERSION
            || header.example_size_bytes != sizeof(TrainingExample)) {
            fclose(f);
            return false;
        }

        examples.resize(header.num_examples);
        bool ok = fread(examples.data(), sizeof(TrainingExample), header.num_examples, f)
                  == header.num_examples;

        fclose(f);
        return ok;
    }
};

// --- Neural network evaluator ---

// Bridges the DC0Network to the EvalFunction interface used by MCTS.
// Performs single-position inference (not batched — batching is a Phase 6 optimization).
class NNEvaluator {
public:
    NNEvaluator(DC0Network model, torch::Device device)
        : model_(model), device_(device) {
        model_->to(device_);
        model_->eval();
    }

    // EvalFunction-compatible: given a Board, return (policy_logits, value).
    std::pair<std::vector<float>, float> evaluate(const Board& board) {
        // Encode board
        float planes[ENCODING_SIZE];
        encode_board(board, planes);

        // Create tensor and move to device
        auto input = torch::from_blob(planes, {1, ENCODING_PLANES, 8, 8},
                                      torch::kFloat32).clone().to(device_);

        // Encode legal moves for masking
        // encode_legal_moves modifies nothing on the Board, but takes non-const ref
        // because it reads board.state.moves which is mutable.
        // We need a mutable board reference — cast away const since we know it's safe.
        bool legal_mask_arr[POLICY_SIZE];
        encode_legal_moves(const_cast<Board&>(board), legal_mask_arr);

        // Create legal move mask tensor
        auto mask_tensor = torch::zeros({1, POLICY_SIZE}, torch::kBool);
        auto mask_acc = mask_tensor.accessor<bool, 2>();
        for (int i = 0; i < POLICY_SIZE; ++i) {
            mask_acc[0][i] = legal_mask_arr[i];
        }
        mask_tensor = mask_tensor.to(device_);

        // Forward pass (no grad)
        torch::NoGradGuard no_grad;
        auto [policy, wdl] = model_->predict(input, mask_tensor);

        // policy: (1, 4672) probabilities after softmax+masking
        // wdl: (1, 3) probabilities [P(win), P(draw), P(loss)]

        // Convert policy to vector of logits (actually probabilities, but MCTS
        // will softmax them again — so convert back to log space)
        auto policy_cpu = policy.squeeze(0).to(torch::kCPU);
        auto policy_data = policy_cpu.data_ptr<float>();

        std::vector<float> policy_vec(POLICY_SIZE);
        for (int i = 0; i < POLICY_SIZE; ++i) {
            // Convert probability back to logit for MCTS expand() which does its own softmax
            float p = policy_data[i];
            policy_vec[i] = (p > 1e-8f) ? std::log(p) : -30.0f;
        }

        // Convert WDL to scalar value
        auto wdl_cpu = wdl.squeeze(0).to(torch::kCPU);
        float w = wdl_cpu[0].item<float>();
        float l = wdl_cpu[2].item<float>();
        float value = w - l;

        return {policy_vec, value};
    }

    // Get the EvalFunction for MCTS (single-position, unbatched)
    EvalFunction get_eval_function() {
        return [this](const Board& board) -> std::pair<std::vector<float>, float> {
            return evaluate(board);
        };
    }

    // Batched evaluation: encode N pre-encoded boards in a single GPU call.
    // planes_batch: N * ENCODING_SIZE floats (already encoded by caller)
    // masks_batch:  N * POLICY_SIZE bools (already encoded by caller)
    // out_policies: N * POLICY_SIZE floats (logits for MCTS expand softmax)
    // out_values:   N floats (scalar value in [-1, 1])
    void evaluate_batch(
        const float* planes_batch,
        const bool* masks_batch,
        int n,
        float* out_policies,
        float* out_values
    ) {
        // Build input tensor: (N, ENCODING_PLANES, 8, 8)
        auto input = torch::from_blob(
            const_cast<float*>(planes_batch),
            {n, ENCODING_PLANES, 8, 8}, torch::kFloat32
        ).clone().to(device_);

        // Build legal mask tensor: (N, POLICY_SIZE)
        auto mask_tensor = torch::zeros({n, POLICY_SIZE}, torch::kBool);
        auto mask_acc = mask_tensor.accessor<bool, 2>();
        for (int i = 0; i < n; ++i) {
            const bool* row = &masks_batch[i * POLICY_SIZE];
            for (int j = 0; j < POLICY_SIZE; ++j) {
                mask_acc[i][j] = row[j];
            }
        }
        mask_tensor = mask_tensor.to(device_);

        // Single forward pass for the whole batch
        torch::NoGradGuard no_grad;
        auto [policy, wdl] = model_->predict(input, mask_tensor);

        // policy: (N, POLICY_SIZE) probabilities
        // wdl:    (N, 3) probabilities
        auto policy_cpu = policy.to(torch::kCPU).contiguous();
        auto wdl_cpu = wdl.to(torch::kCPU).contiguous();
        auto* p_data = policy_cpu.data_ptr<float>();
        auto* w_data = wdl_cpu.data_ptr<float>();

        for (int i = 0; i < n; ++i) {
            // Convert probabilities back to logits for MCTS expand() softmax
            const float* p_row = &p_data[i * POLICY_SIZE];
            float* out_row = &out_policies[i * POLICY_SIZE];
            for (int j = 0; j < POLICY_SIZE; ++j) {
                out_row[j] = (p_row[j] > 1e-8f) ? std::log(p_row[j]) : -30.0f;
            }
            // Convert WDL to scalar value
            out_values[i] = w_data[i * 3 + 0] - w_data[i * 3 + 2];
        }
    }

    // Get the BatchedEvalFunction for batched MCTS
    BatchedEvalFunction get_batched_eval_function() {
        return [this](const float* planes, const bool* masks, int n,
                      float* out_policies, float* out_values) {
            evaluate_batch(planes, masks, n, out_policies, out_values);
        };
    }

    DC0Network& model() { return model_; }

private:
    DC0Network model_;
    torch::Device device_;
};

// --- Self-play game ---

struct SelfPlayConfig {
    int simulations_per_move = 800;    // MCTS simulations per move
    float c_puct = 2.5f;              // PUCT exploration constant
    float dirichlet_alpha = 0.3f;     // Dirichlet noise alpha
    float dirichlet_epsilon = 0.25f;  // Dirichlet noise weight
    int temperature_moves = 30;       // moves with temperature=1, then temp=0
    float temperature = 1.0f;         // initial temperature
    int max_game_moves = 512;         // maximum moves per game before forced draw
    int batch_size = 64;              // batch size for batched MCTS inference
};

// Result of a single self-play game
struct GameResult {
    int num_moves = 0;
    float outcome = 0.0f;  // from white's perspective: 1=white wins, -1=black wins, 0=draw
    bool is_checkmate = false;
    bool is_draw = false;
};

// Play a single self-play game and collect training examples.
// Returns the game result. Appends training examples to `data`.
inline GameResult play_self_play_game(
    const EvalFunction& eval_fn,
    const SelfPlayConfig& config,
    TrainingData& data
) {
    fen::FEN start = fen::load_from_string(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Board board(start);

    MCTSTree tree;
    tree.c_puct = config.c_puct;
    tree.dirichlet_alpha = config.dirichlet_alpha;
    tree.dirichlet_epsilon = config.dirichlet_epsilon;
    tree.add_noise = true;
    tree.set_root(board);

    // Temporary storage for positions before game result is known
    struct PendingExample {
        float planes[ENCODING_SIZE];
        float policy[POLICY_SIZE];
        COLOR side_to_move;
    };
    std::vector<PendingExample> pending;
    pending.reserve(config.max_game_moves);

    GameResult result;

    for (int move_num = 0; move_num < config.max_game_moves; ++move_num) {
        if (board.is_checkmate()) {
            // Side to move lost
            result.is_checkmate = true;
            result.outcome = (board.activePlayer() == WHITE) ? -1.0f : 1.0f;
            break;
        }
        if (board.is_draw()) {
            result.is_draw = true;
            result.outcome = 0.0f;
            break;
        }

        // Run MCTS
        tree.run_simulations(config.simulations_per_move, eval_fn);

        if (move_num > 0 && move_num % 50 == 0) {
            DC0_LOG_DEBUG("  game move %d ...", move_num);
        }

        // Record position + MCTS policy
        PendingExample pe;
        encode_board(board, pe.planes);
        float temp = (move_num < config.temperature_moves) ? config.temperature : 0.0f;
        tree.get_policy(pe.policy, (move_num < config.temperature_moves) ? 1.0f : 0.0f);
        pe.side_to_move = board.activePlayer();
        pending.push_back(pe);

        // Select move
        move_t m = tree.select_move(temp);
        if (m == board::nullmove) {
            // Shouldn't happen if board isn't terminal, but handle gracefully
            result.is_draw = true;
            result.outcome = 0.0f;
            break;
        }

        // Make the move
        board.make_move(m);
        tree.advance(m, board);
        result.num_moves++;
    }

    // If we hit max moves, treat as draw
    if (!result.is_checkmate && !result.is_draw) {
        result.is_draw = true;
        result.outcome = 0.0f;
    }

    // Convert pending examples to training data with correct results
    for (auto& pe : pending) {
        TrainingExample ex;
        std::memcpy(ex.planes, pe.planes, sizeof(ex.planes));
        std::memcpy(ex.policy, pe.policy, sizeof(ex.policy));
        // Result from this player's perspective
        if (pe.side_to_move == WHITE) {
            ex.result = result.outcome;
        } else {
            ex.result = -result.outcome;
        }
        data.add(ex);
    }

    return result;
}

// Play a single self-play game using batched MCTS inference.
// Same interface as play_self_play_game but uses run_simulations_batched.
inline GameResult play_self_play_game_batched(
    const BatchedEvalFunction& batch_eval_fn,
    const SelfPlayConfig& config,
    TrainingData& data
) {
    fen::FEN start = fen::load_from_string(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Board board(start);

    MCTSTree tree;
    tree.c_puct = config.c_puct;
    tree.dirichlet_alpha = config.dirichlet_alpha;
    tree.dirichlet_epsilon = config.dirichlet_epsilon;
    tree.add_noise = true;
    tree.set_root(board);

    struct PendingExample {
        float planes[ENCODING_SIZE];
        float policy[POLICY_SIZE];
        COLOR side_to_move;
    };
    std::vector<PendingExample> pending;
    pending.reserve(config.max_game_moves);

    GameResult result;

    for (int move_num = 0; move_num < config.max_game_moves; ++move_num) {
        if (board.is_checkmate()) {
            result.is_checkmate = true;
            result.outcome = (board.activePlayer() == WHITE) ? -1.0f : 1.0f;
            break;
        }
        if (board.is_draw()) {
            result.is_draw = true;
            result.outcome = 0.0f;
            break;
        }

        // Run batched MCTS
        tree.run_simulations_batched(
            config.simulations_per_move, batch_eval_fn, config.batch_size);

        if (move_num > 0 && move_num % 50 == 0) {
            DC0_LOG_DEBUG("  game move %d ...", move_num);
        }

        PendingExample pe;
        encode_board(board, pe.planes);
        float temp = (move_num < config.temperature_moves) ? config.temperature : 0.0f;
        tree.get_policy(pe.policy, (move_num < config.temperature_moves) ? 1.0f : 0.0f);
        pe.side_to_move = board.activePlayer();
        pending.push_back(pe);

        move_t m = tree.select_move(temp);
        if (m == board::nullmove) {
            result.is_draw = true;
            result.outcome = 0.0f;
            break;
        }

        board.make_move(m);
        tree.advance(m, board);
        result.num_moves++;
    }

    if (!result.is_checkmate && !result.is_draw) {
        result.is_draw = true;
        result.outcome = 0.0f;
    }

    for (auto& pe : pending) {
        TrainingExample ex;
        std::memcpy(ex.planes, pe.planes, sizeof(ex.planes));
        std::memcpy(ex.policy, pe.policy, sizeof(ex.policy));
        if (pe.side_to_move == WHITE) {
            ex.result = result.outcome;
        } else {
            ex.result = -result.outcome;
        }
        data.add(ex);
    }

    return result;
}

// --- Self-play manager ---

struct SelfPlayStats {
    int total_games = 0;
    int total_positions = 0;
    int white_wins = 0;
    int black_wins = 0;
    int draws = 0;
    int total_moves = 0;
    int64_t total_nodes = 0;       // total MCTS simulations across all games
    double elapsed_sec = 0.0;      // wall-clock time for all games

    // Derived metrics
    double nodes_per_sec() const { return elapsed_sec > 0 ? total_nodes / elapsed_sec : 0; }
    double positions_per_sec() const { return elapsed_sec > 0 ? total_positions / elapsed_sec : 0; }
    double sec_per_game() const { return total_games > 0 ? elapsed_sec / total_games : 0; }
};

// Run multiple self-play games and write training data to disk.
inline SelfPlayStats run_self_play(
    NNEvaluator& evaluator,
    int num_games,
    const SelfPlayConfig& config,
    const std::string& output_path
) {
    TrainingData data;
    data.reserve(num_games * 200);  // rough estimate: ~200 positions per game

    SelfPlayStats stats;

    auto batch_eval_fn = evaluator.get_batched_eval_function();
    auto t_start = std::chrono::steady_clock::now();

    for (int game = 0; game < num_games; ++game) {
        auto t_game_start = std::chrono::steady_clock::now();
        GameResult result = play_self_play_game_batched(batch_eval_fn, config, data);
        auto t_game_end = std::chrono::steady_clock::now();
        double game_sec = std::chrono::duration<double>(t_game_end - t_game_start).count();

        stats.total_games++;
        stats.total_moves += result.num_moves;
        // Each move runs simulations_per_move MCTS nodes
        stats.total_nodes += static_cast<int64_t>(result.num_moves) * config.simulations_per_move;
        if (result.outcome > 0.5f) stats.white_wins++;
        else if (result.outcome < -0.5f) stats.black_wins++;
        else stats.draws++;

        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        stats.elapsed_sec = elapsed;
        stats.total_positions = static_cast<int>(data.size());

        const char* outcome_str = result.is_checkmate
            ? (result.outcome > 0 ? "1-0" : "0-1")
            : (result.is_draw ? "1/2" : "???");
        DC0_LOG_INFO("Game %d/%d: %s in %d moves (%.1fs) | "
                "total: %d pos, %.1f pos/s, %.0f nodes/s, %.1f s/game",
                game + 1, num_games, outcome_str, result.num_moves, game_sec,
                stats.total_positions,
                stats.positions_per_sec(), stats.nodes_per_sec(), stats.sec_per_game());
    }

    auto t_end = std::chrono::steady_clock::now();
    stats.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    stats.total_positions = static_cast<int>(data.size());

    DC0_LOG_INFO("Self-play done: %d games, %d positions in %.1f s "
            "(%.1f pos/s, %.0f nodes/s, %.1f s/game)",
            stats.total_games, stats.total_positions, stats.elapsed_sec,
            stats.positions_per_sec(), stats.nodes_per_sec(), stats.sec_per_game());

    // Save to disk
    if (!output_path.empty()) {
        if (data.save(output_path)) {
            DC0_LOG_INFO("Saved %zu training examples to %s",
                    data.size(), output_path.c_str());
        } else {
            DC0_LOG_ERROR("Failed to save training data to %s",
                    output_path.c_str());
        }
    }

    return stats;
}

} // namespace dc0
