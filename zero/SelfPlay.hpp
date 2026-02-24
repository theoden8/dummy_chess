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

        // Forward pass (no grad) — get logits directly, skip softmax->log round-trip
        torch::NoGradGuard no_grad;
        auto [policy_logits, wdl] = model_->predict_logits(input, mask_tensor);

        // policy_logits: (1, 4672) masked logits (illegal = -1e32)
        // wdl: (1, 3) probabilities [P(win), P(draw), P(loss)]

        auto logits_cpu = policy_logits.squeeze(0).to(torch::kCPU);
        auto logits_data = logits_cpu.data_ptr<float>();

        std::vector<float> policy_vec(logits_data, logits_data + POLICY_SIZE);

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

        // Build legal mask tensor from bool array using memcpy + conversion.
        // torch has no bool from_blob that works reliably across platforms,
        // so convert through uint8 which has the same layout as bool.
        auto mask_u8 = torch::from_blob(
            const_cast<bool*>(masks_batch),
            {n, POLICY_SIZE}, torch::kBool
        ).clone().to(device_);

        // Single forward pass — get logits directly (no softmax->log round-trip)
        torch::NoGradGuard no_grad;
        auto [policy_logits, wdl] = model_->predict_logits(input, mask_u8);

        // policy_logits: (N, POLICY_SIZE) masked logits
        // wdl: (N, 3) probabilities
        auto logits_cpu = policy_logits.to(torch::kCPU).contiguous();
        auto wdl_cpu = wdl.to(torch::kCPU).contiguous();
        auto* p_data = logits_cpu.data_ptr<float>();
        auto* w_data = wdl_cpu.data_ptr<float>();

        // Copy logits directly — no per-element log() needed
        std::memcpy(out_policies, p_data, n * POLICY_SIZE * sizeof(float));

        for (int i = 0; i < n; ++i) {
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
    int n_parallel = 8;               // games in flight for parallel self-play
};

// Result of a single self-play game
struct GameResult {
    int num_moves = 0;
    float outcome = 0.0f;  // from white's perspective: 1=white wins, -1=black wins, 0=draw
    bool is_checkmate = false;
    bool is_draw = false;

    // Metrics accumulated during the game
    float sum_policy_entropy = 0.0f;   // sum of -sum(p*log(p)) over moves
    float sum_root_q = 0.0f;           // sum of abs(root Q) over moves
    float sum_top_prior = 0.0f;        // sum of NN prior of the move MCTS picked

    float avg_policy_entropy() const { return num_moves > 0 ? sum_policy_entropy / num_moves : 0.0f; }
    float avg_root_q_abs() const { return num_moves > 0 ? sum_root_q / num_moves : 0.0f; }
    float avg_top_prior() const { return num_moves > 0 ? sum_top_prior / num_moves : 0.0f; }
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

        // Collect metrics from the MCTS search
        {
            auto stats = tree.get_move_stats();
            // Policy entropy: -sum(p * log(p)) where p is the visit distribution
            float entropy = 0.0f;
            uint32_t total_visits = 0;
            for (auto& s : stats) total_visits += s.visits;
            if (total_visits > 0) {
                for (auto& s : stats) {
                    if (s.visits == 0) continue;
                    float p = float(s.visits) / float(total_visits);
                    entropy -= p * std::log(p);
                }
            }
            result.sum_policy_entropy += entropy;

            // Root Q: average Q of best move (by visits)
            float best_q = 0.0f;
            uint32_t best_visits = 0;
            float selected_prior = 0.0f;
            move_t selected = tree.select_move(temp);
            for (auto& s : stats) {
                if (s.visits > best_visits) {
                    best_visits = s.visits;
                    best_q = s.q_value;
                }
                if (s.move == selected) {
                    selected_prior = s.prior;
                }
            }
            result.sum_root_q += std::abs(best_q);
            result.sum_top_prior += selected_prior;
        }

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

    // Accumulated quality metrics
    float sum_policy_entropy = 0.0f;
    float sum_root_q_abs = 0.0f;
    float sum_top_prior = 0.0f;
    int metric_count = 0;          // total moves across all games for averaging

    // Derived metrics
    double nodes_per_sec() const { return elapsed_sec > 0 ? total_nodes / elapsed_sec : 0; }
    double positions_per_sec() const { return elapsed_sec > 0 ? total_positions / elapsed_sec : 0; }
    double sec_per_game() const { return total_games > 0 ? elapsed_sec / total_games : 0; }
    float avg_policy_entropy() const { return metric_count > 0 ? sum_policy_entropy / metric_count : 0.0f; }
    float avg_root_q_abs() const { return metric_count > 0 ? sum_root_q_abs / metric_count : 0.0f; }
    float avg_top_prior() const { return metric_count > 0 ? sum_top_prior / metric_count : 0.0f; }
    float avg_game_length() const { return total_games > 0 ? float(total_moves) / total_games : 0.0f; }
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

        // Accumulate quality metrics
        stats.sum_policy_entropy += result.sum_policy_entropy;
        stats.sum_root_q_abs += result.sum_root_q;
        stats.sum_top_prior += result.sum_top_prior;
        stats.metric_count += result.num_moves;

        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        stats.elapsed_sec = elapsed;
        stats.total_positions = static_cast<int>(data.size());

        const char* outcome_str = result.is_checkmate
            ? (result.outcome > 0 ? "1-0" : "0-1")
            : (result.is_draw ? "1/2" : "???");
        DC0_LOG_INFO("Game %d/%d: %s in %d moves (%.1fs) | "
                "entropy=%.2f |Q|=%.3f prior=%.3f | "
                "%.0f nodes/s, %.1f s/game",
                game + 1, num_games, outcome_str, result.num_moves, game_sec,
                result.avg_policy_entropy(), result.avg_root_q_abs(), result.avg_top_prior(),
                stats.nodes_per_sec(), stats.sec_per_game());
    }

    auto t_end = std::chrono::steady_clock::now();
    stats.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    stats.total_positions = static_cast<int>(data.size());

    DC0_LOG_INFO("Self-play done: %d games, %d positions in %.1f s "
            "(%.1f pos/s, %.0f nodes/s, %.1f s/game)",
            stats.total_games, stats.total_positions, stats.elapsed_sec,
            stats.positions_per_sec(), stats.nodes_per_sec(), stats.sec_per_game());
    DC0_LOG_INFO("Quality: entropy=%.2f |Q|=%.3f prior=%.3f avg_len=%.0f "
            "W/D/L=%d/%d/%d",
            stats.avg_policy_entropy(), stats.avg_root_q_abs(), stats.avg_top_prior(),
            stats.avg_game_length(),
            stats.white_wins, stats.draws, stats.black_wins);

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

// --- Parallel multi-game self-play ---
// Plays N games concurrently, batching NN evaluations across all game trees.
// This keeps the GPU busy by filling batches from multiple games.

inline SelfPlayStats run_self_play_parallel(
    NNEvaluator& evaluator,
    int num_games,
    const SelfPlayConfig& config,
    const std::string& output_path,
    int n_parallel = 8
) {
    TrainingData data;
    data.reserve(num_games * 200);

    SelfPlayStats stats;
    auto batch_eval_fn = evaluator.get_batched_eval_function();
    auto t_start = std::chrono::steady_clock::now();

    static const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // Per-position pending example (mirrors the sequential version's struct).
    struct PendingExample {
        float planes[ENCODING_SIZE];
        float policy[POLICY_SIZE];
        COLOR side_to_move;
    };

    // State for each active game slot.
    // Board has deleted copy-assignment (const members), so we use unique_ptr.
    struct Slot {
        MCTSTree tree;
        std::unique_ptr<Board> board;
        std::vector<PendingExample> pending;
        GameResult result;
        int sims_this_move = 0;
        int move_num = 0;
        bool active = false;
        int game_id = -1;

        void start_new_game(int id, const SelfPlayConfig& cfg) {
            board = std::make_unique<Board>(fen::load_from_string(START_FEN));
            tree = MCTSTree();
            tree.c_puct = cfg.c_puct;
            tree.dirichlet_alpha = cfg.dirichlet_alpha;
            tree.dirichlet_epsilon = cfg.dirichlet_epsilon;
            tree.add_noise = true;
            tree.set_root(*board);
            pending.clear();
            result = GameResult();
            sims_this_move = 0;
            move_num = 0;
            active = true;
            game_id = id;
        }
    };

    std::vector<Slot> slots(n_parallel);
    int next_game_id = 0;
    int games_completed = 0;

    // Finalize a completed game: convert pending examples, update stats,
    // log, and either start a new game or deactivate the slot.
    auto finalize_game = [&](Slot& slot) {
        // Convert pending examples to training data
        for (auto& pe : slot.pending) {
            TrainingExample ex;
            std::memcpy(ex.planes, pe.planes, sizeof(ex.planes));
            std::memcpy(ex.policy, pe.policy, sizeof(ex.policy));
            if (pe.side_to_move == WHITE) {
                ex.result = slot.result.outcome;
            } else {
                ex.result = -slot.result.outcome;
            }
            data.add(ex);
        }

        // Update stats
        games_completed++;
        stats.total_games++;
        stats.total_moves += slot.result.num_moves;
        stats.total_nodes += static_cast<int64_t>(slot.result.num_moves)
                             * config.simulations_per_move;
        if (slot.result.outcome > 0.5f) stats.white_wins++;
        else if (slot.result.outcome < -0.5f) stats.black_wins++;
        else stats.draws++;

        stats.sum_policy_entropy += slot.result.sum_policy_entropy;
        stats.sum_root_q_abs += slot.result.sum_root_q;
        stats.sum_top_prior += slot.result.sum_top_prior;
        stats.metric_count += slot.result.num_moves;

        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        stats.elapsed_sec = elapsed;
        stats.total_positions = static_cast<int>(data.size());

        const char* outcome_str = slot.result.is_checkmate
            ? (slot.result.outcome > 0 ? "1-0" : "0-1")
            : (slot.result.is_draw ? "1/2" : "???");
        DC0_LOG_INFO("Game %d/%d: %s in %d moves | "
                "entropy=%.2f |Q|=%.3f prior=%.3f | "
                "%.0f nodes/s, %.1f s/game",
                games_completed, num_games, outcome_str, slot.result.num_moves,
                slot.result.avg_policy_entropy(), slot.result.avg_root_q_abs(),
                slot.result.avg_top_prior(),
                stats.nodes_per_sec(), stats.sec_per_game());

        // Start next game or deactivate slot
        if (next_game_id < num_games) {
            slot.start_new_game(next_game_id++, config);
            slot.tree.ensure_root_expanded(batch_eval_fn);
        } else {
            slot.active = false;
        }
    };

    // Initialize slots
    for (int i = 0; i < n_parallel && next_game_id < num_games; ++i) {
        slots[i].start_new_game(next_game_id++, config);
    }

    // Expand all roots
    for (auto& slot : slots) {
        if (slot.active) {
            slot.tree.ensure_root_expanded(batch_eval_fn);
        }
    }

    // Shared batch buffers
    const int max_batch = config.batch_size;
    std::vector<float> planes_buf(max_batch * ENCODING_SIZE);
    std::unique_ptr<bool[]> masks_buf(new bool[max_batch * POLICY_SIZE]);
    std::vector<float> policies_buf(max_batch * POLICY_SIZE);
    std::vector<float> values_buf(max_batch);

    // Main loop: keep running until all games are done
    while (games_completed < num_games) {
        // Phase 0: Finalize any games whose roots are already terminal
        // (e.g. after advance() lands on checkmate/draw).
        for (auto& slot : slots) {
            if (!slot.active) continue;
            if (slot.board->is_checkmate()) {
                slot.result.is_checkmate = true;
                slot.result.outcome = (slot.board->activePlayer() == WHITE) ? -1.0f : 1.0f;
                finalize_game(slot);
            } else if (slot.board->is_draw()) {
                slot.result.is_draw = true;
                slot.result.outcome = 0.0f;
                finalize_game(slot);
            }
        }

        // Check if all games done after terminal finalization
        if (games_completed >= num_games) break;

        // Phase 1: Collect leaves from all active games
        std::vector<MCTSTree::PendingLeaf> pending_leaves;
        pending_leaves.reserve(max_batch);
        int nn_count = 0;

        // Collect multiple leaves per game to fill the batch
        int active_count = 0;
        for (auto& slot : slots) {
            if (slot.active) active_count++;
        }
        int leaves_per_game = active_count > 0
            ? std::max(1, max_batch / active_count) : 1;

        for (int slot_idx = 0; slot_idx < n_parallel; ++slot_idx) {
            auto& slot = slots[slot_idx];
            if (!slot.active) continue;
            if (slot.tree.is_terminal()) continue;

            int remaining_sims = config.simulations_per_move - slot.sims_this_move;
            int n_leaves = std::min(leaves_per_game, remaining_sims);
            n_leaves = std::min(n_leaves, max_batch - (int)pending_leaves.size());

            for (int l = 0; l < n_leaves; ++l) {
                auto leaf = slot.tree.select_leaf();
                leaf.game_index = slot_idx;
                if (leaf.needs_nn_eval) {
                    std::memcpy(&planes_buf[nn_count * ENCODING_SIZE],
                                leaf.planes, ENCODING_SIZE * sizeof(float));
                    std::memcpy(&masks_buf[nn_count * POLICY_SIZE],
                                leaf.legal_mask, POLICY_SIZE * sizeof(bool));
                    leaf.batch_index = nn_count;
                    nn_count++;
                }
                pending_leaves.push_back(std::move(leaf));
            }

            if ((int)pending_leaves.size() >= max_batch) break;
        }

        // If no leaves collected, all remaining active games must have
        // terminal trees. They'll be caught by Phase 0 on the next iteration.
        if (pending_leaves.empty()) continue;

        // Phase 2: Batch NN evaluation
        if (nn_count > 0) {
            batch_eval_fn(planes_buf.data(), masks_buf.get(), nn_count,
                          policies_buf.data(), values_buf.data());
        }

        // Phase 3: Process leaves — expand and backpropagate
        for (auto& leaf : pending_leaves) {
            auto& slot = slots[leaf.game_index];
            if (leaf.needs_nn_eval) {
                float* policy = &policies_buf[leaf.batch_index * POLICY_SIZE];
                float value = values_buf[leaf.batch_index];
                slot.tree.process_leaf(leaf, policy, value);
            } else {
                slot.tree.process_leaf(leaf, nullptr, leaf.terminal_value);
            }
            slot.sims_this_move++;
        }

        // Phase 4: Check which games have finished their simulations for the current move
        for (int slot_idx = 0; slot_idx < n_parallel; ++slot_idx) {
            auto& slot = slots[slot_idx];
            if (!slot.active) continue;

            // Not enough sims yet for this move
            if (slot.sims_this_move < config.simulations_per_move) continue;

            // Sims done for this move — record example and pick move
            float temp = (slot.move_num < config.temperature_moves) ? config.temperature : 0.0f;

            // Record training example
            PendingExample pe;
            encode_board(*slot.board, pe.planes);
            slot.tree.get_policy(pe.policy,
                (slot.move_num < config.temperature_moves) ? 1.0f : 0.0f);
            pe.side_to_move = slot.board->activePlayer();
            slot.pending.push_back(pe);

            // Collect metrics
            {
                auto mstats = slot.tree.get_move_stats();
                float entropy = 0.0f;
                uint32_t total_visits = 0;
                for (auto& s : mstats) total_visits += s.visits;
                if (total_visits > 0) {
                    for (auto& s : mstats) {
                        if (s.visits == 0) continue;
                        float p = float(s.visits) / float(total_visits);
                        entropy -= p * std::log(p);
                    }
                }
                slot.result.sum_policy_entropy += entropy;

                float best_q = 0.0f;
                uint32_t best_visits = 0;
                for (auto& s : mstats) {
                    if (s.visits > best_visits) {
                        best_visits = s.visits;
                        best_q = s.q_value;
                    }
                }
                slot.result.sum_root_q += std::abs(best_q);
            }

            // Pick and play move
            move_t m = slot.tree.select_move(temp);
            if (m == board::nullmove) {
                slot.result.is_draw = true;
                slot.result.outcome = 0.0f;
                finalize_game(slot);
                continue;
            }

            // Record prior of selected move
            {
                auto mstats = slot.tree.get_move_stats();
                for (auto& s : mstats) {
                    if (s.move == m) {
                        slot.result.sum_top_prior += s.prior;
                        break;
                    }
                }
            }

            slot.board->make_move(m);
            slot.tree.advance(m, *slot.board);
            slot.result.num_moves++;
            slot.move_num++;
            slot.sims_this_move = 0;

            // Expand new root
            if (!slot.tree.is_terminal()) {
                slot.tree.ensure_root_expanded(batch_eval_fn);
            }

            // Check move limit
            if (slot.move_num >= config.max_game_moves) {
                slot.result.is_draw = true;
                slot.result.outcome = 0.0f;
                finalize_game(slot);
                continue;
            }

            // Terminal positions after the move are caught by Phase 0
            // on the next iteration of the main loop.
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    stats.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    stats.total_positions = static_cast<int>(data.size());

    DC0_LOG_INFO("Self-play done: %d games, %d positions in %.1f s "
            "(%.1f pos/s, %.0f nodes/s, %.1f s/game) [%d parallel]",
            stats.total_games, stats.total_positions, stats.elapsed_sec,
            stats.positions_per_sec(), stats.nodes_per_sec(), stats.sec_per_game(),
            n_parallel);
    DC0_LOG_INFO("Quality: entropy=%.2f |Q|=%.3f prior=%.3f avg_len=%.0f "
            "W/D/L=%d/%d/%d",
            stats.avg_policy_entropy(), stats.avg_root_q_abs(), stats.avg_top_prior(),
            stats.avg_game_length(),
            stats.white_wins, stats.draws, stats.black_wins);

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
