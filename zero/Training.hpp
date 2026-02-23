#pragma once

// dc0 training loop: train the neural network from self-play data.
//
// Components:
//   - TrainingDataset: libtorch Dataset wrapping TrainingData
//   - TrainingConfig: hyperparameters for training
//   - Trainer: trains DC0Network with SGD, saves checkpoints
//   - evaluate_models: pit two models against each other

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <filesystem>

#include <torch/torch.h>

#include <m42.h>
#include <Zobrist.hpp>
#include <Board.hpp>
#include <FEN.hpp>

#include <MoveEncoding.hpp>
#include <BoardEncoding.hpp>
#include <Network.hpp>
#include <MCTS.hpp>
#include <SelfPlay.hpp>
#include <Logging.hpp>

namespace dc0 {

// --- Training dataset ---

// libtorch Dataset that wraps our TrainingData binary format.
// Returns tensors suitable for training: (input, policy_target, value_target).
class TrainingDataset : public torch::data::datasets::Dataset<TrainingDataset> {
public:
    TrainingDataset() = default;

    // Load from a single binary file
    explicit TrainingDataset(const std::string& path) {
        data_.load(path);
    }

    // Load from TrainingData directly
    explicit TrainingDataset(const TrainingData& data) : data_(data) {}

    // Append data from a file
    void append_file(const std::string& path) {
        TrainingData more;
        if (more.load(path)) {
            data_.examples.insert(data_.examples.end(),
                                  more.examples.begin(), more.examples.end());
        }
    }

    // Append data from another TrainingData
    void append(const TrainingData& other) {
        data_.examples.insert(data_.examples.end(),
                              other.examples.begin(), other.examples.end());
    }

    // Dataset interface
    torch::data::Example<> get(size_t index) override {
        const auto& ex = data_.examples[index];

        // Input: (22, 8, 8) float tensor
        auto input = torch::from_blob(
            const_cast<float*>(ex.planes),
            {ENCODING_PLANES, 8, 8}, torch::kFloat32
        ).clone();

        // Target: policy (4672) + value (1 float, but stored as index into {win, draw, loss})
        // We pack both into a single target tensor for convenience:
        // target[0:4672] = policy, target[4672] = result
        auto target = torch::zeros({POLICY_SIZE + 1});
        auto target_acc = target.accessor<float, 1>();
        for (int i = 0; i < POLICY_SIZE; ++i) {
            target_acc[i] = ex.policy[i];
        }
        target_acc[POLICY_SIZE] = ex.result;

        return {input, target};
    }

    torch::optional<size_t> size() const override {
        return data_.examples.size();
    }

    size_t num_examples() const { return data_.examples.size(); }

private:
    TrainingData data_;
};

// --- Training configuration ---

struct TrainingConfig {
    // Optimizer
    float learning_rate = 0.02f;
    float momentum = 0.9f;
    float weight_decay = 1e-4f;

    // Schedule: LR drops at these step counts
    std::vector<int> lr_steps = {100000, 300000};
    float lr_gamma = 0.1f;  // LR multiplied by this at each step

    // Training
    int batch_size = 256;
    int epochs = 1;          // epochs over the current data window
    int log_interval = 100;  // print loss every N batches

    // Loss weights
    float policy_weight = 1.0f;
    float value_weight = 1.0f;

    // Data
    int max_examples = 500000;  // sliding window size
};

// --- Training result ---

struct TrainingResult {
    float avg_policy_loss = 0.0f;
    float avg_value_loss = 0.0f;
    float avg_total_loss = 0.0f;
    int total_batches = 0;
    int total_examples = 0;
    double elapsed_sec = 0.0;      // wall-clock time for training

    // Derived metrics
    double examples_per_sec() const { return elapsed_sec > 0 ? total_examples / elapsed_sec : 0; }
    double batches_per_sec() const { return elapsed_sec > 0 ? total_batches / elapsed_sec : 0; }
};

// --- Trainer ---

class Trainer {
public:
    Trainer(DC0Network model, torch::Device device, const TrainingConfig& config)
        : model_(model), device_(device), config_(config), global_step_(0)
    {
        model_->to(device_);

        // Create SGD optimizer
        optimizer_ = std::make_unique<torch::optim::SGD>(
            model_->parameters(),
            torch::optim::SGDOptions(config.learning_rate)
                .momentum(config.momentum)
                .weight_decay(config.weight_decay)
        );
    }

    // Train on a dataset for the configured number of epochs.
    TrainingResult train(TrainingDataset& dataset) {
        model_->train();

        auto data_loader = torch::data::make_data_loader(
            dataset.map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions()
                .batch_size(config_.batch_size)
                .workers(2)
        );

        TrainingResult result;
        auto t_start = std::chrono::steady_clock::now();

        for (int epoch = 0; epoch < config_.epochs; ++epoch) {
            for (auto& batch : *data_loader) {
                auto inputs = batch.data.to(device_);
                auto targets = batch.target.to(device_);

                // Split target into policy and value
                auto policy_target = targets.slice(/*dim=*/1, 0, POLICY_SIZE);
                auto value_target = targets.select(/*dim=*/1, POLICY_SIZE);

                // Forward pass
                auto [policy_logits, wdl_logits] = model_->forward(inputs);

                // Policy loss: cross-entropy with MCTS visit distribution
                // policy_target is a probability distribution, use KL-div-like loss:
                // L = -sum(pi * log(softmax(logits))) = cross-entropy
                auto log_policy = torch::log_softmax(policy_logits, /*dim=*/1);
                auto policy_loss = -(policy_target * log_policy).sum(/*dim=*/1).mean();

                // Value loss: cross-entropy with WDL target
                // Convert scalar result to WDL target: result=1 -> (1,0,0), result=0 -> (0,1,0), result=-1 -> (0,0,1)
                // For intermediate values (shouldn't happen but handle gracefully):
                //   WDL target = ((1+z)/2, 0, (1-z)/2) clamped
                auto wdl_target = torch::zeros({value_target.size(0), 3}, device_);
                wdl_target.select(1, 0) = (value_target + 1.0f) / 2.0f;  // P(win)
                wdl_target.select(1, 2) = (1.0f - value_target) / 2.0f;  // P(loss)
                // P(draw) = 1 - P(win) - P(loss)
                wdl_target.select(1, 1) = 1.0f - wdl_target.select(1, 0) - wdl_target.select(1, 2);

                auto log_wdl = torch::log_softmax(wdl_logits, /*dim=*/1);
                auto value_loss = -(wdl_target * log_wdl).sum(/*dim=*/1).mean();

                // Total loss
                auto total_loss = config_.policy_weight * policy_loss
                                + config_.value_weight * value_loss;

                // Backward + step
                optimizer_->zero_grad();
                total_loss.backward();
                optimizer_->step();

                global_step_++;
                update_lr();

                float pl = policy_loss.item<float>();
                float vl = value_loss.item<float>();
                float tl = total_loss.item<float>();

                result.avg_policy_loss += pl;
                result.avg_value_loss += vl;
                result.avg_total_loss += tl;
                result.total_batches++;
                result.total_examples += static_cast<int>(inputs.size(0));

                if (config_.log_interval > 0 && result.total_batches % config_.log_interval == 0) {
                    auto t_now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                    double ex_per_sec = elapsed > 0 ? result.total_examples / elapsed : 0;
                    DC0_LOG_INFO("step %d batch %d: policy_loss=%.4f value_loss=%.4f "
                            "total=%.4f lr=%.6f | %.0f ex/s",
                            global_step_, result.total_batches, pl, vl, tl,
                            current_lr(), ex_per_sec);
                }
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        result.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

        if (result.total_batches > 0) {
            result.avg_policy_loss /= result.total_batches;
            result.avg_value_loss /= result.total_batches;
            result.avg_total_loss /= result.total_batches;
        }

        DC0_LOG_INFO("Training done: %d examples, %d batches in %.1f s (%.0f ex/s, %.1f batch/s)",
                result.total_examples, result.total_batches, result.elapsed_sec,
                result.examples_per_sec(), result.batches_per_sec());

        return result;
    }

    // Save model checkpoint
    void save_checkpoint(const std::string& path) {
        model_->save_weights(path);
    }

    // Load model checkpoint
    void load_checkpoint(const std::string& path) {
        model_->load_weights(path);
    }

    int global_step() const { return global_step_; }
    float current_lr() const {
        float lr = config_.learning_rate;
        for (int s : config_.lr_steps) {
            if (global_step_ >= s) lr *= config_.lr_gamma;
        }
        return lr;
    }

    DC0Network& model() { return model_; }

private:
    DC0Network model_;
    torch::Device device_;
    TrainingConfig config_;
    std::unique_ptr<torch::optim::SGD> optimizer_;
    int global_step_;

    void update_lr() {
        float lr = current_lr();
        for (auto& group : optimizer_->param_groups()) {
            static_cast<torch::optim::SGDOptions&>(group.options()).lr(lr);
        }
    }
};

// --- Model evaluation: pit two models against each other ---

struct EvalResult {
    int games = 0;
    int new_wins = 0;
    int old_wins = 0;
    int draws = 0;
    float win_rate = 0.0f;  // new model's win rate
};

// Play evaluation games between two models.
// new_model plays white for half the games, black for half.
//
// Note: uses unbatched MCTS because each leaf expansion routes to one of two
// models based on board.activePlayer(). Batching would require splitting the
// batch by model, adding complexity for minimal gain (eval games are few and
// use low sim counts). Self-play — the bottleneck — already uses batched MCTS.
inline EvalResult evaluate_models(
    DC0Network new_model,
    DC0Network old_model,
    torch::Device device,
    int num_games,
    int simulations_per_move = 100
) {
    NNEvaluator new_eval(new_model, device);
    NNEvaluator old_eval(old_model, device);

    SelfPlayConfig config;
    config.simulations_per_move = simulations_per_move;
    config.temperature_moves = 0;  // deterministic play for evaluation
    config.temperature = 0.0f;
    config.max_game_moves = 256;

    EvalResult result;
    result.games = num_games;

    for (int game = 0; game < num_games; ++game) {
        bool new_is_white = (game % 2 == 0);

        // Create eval function that routes to the correct model based on side to move
        auto eval_fn = [&](const Board& board) -> std::pair<std::vector<float>, float> {
            bool is_white = (board.activePlayer() == WHITE);
            if ((is_white && new_is_white) || (!is_white && !new_is_white)) {
                return new_eval.evaluate(board);
            } else {
                return old_eval.evaluate(board);
            }
        };

        // Play the game
        fen::FEN start = fen::load_from_string(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        Board board(start);

        MCTSTree tree;
        tree.c_puct = config.c_puct;
        tree.add_noise = false;
        tree.set_root(board);

        int moves_played = 0;
        float outcome = 0.0f;
        bool game_over = false;

        while (moves_played < config.max_game_moves) {
            if (board.is_checkmate()) {
                outcome = (board.activePlayer() == WHITE) ? -1.0f : 1.0f;
                game_over = true;
                break;
            }
            if (board.is_draw()) {
                outcome = 0.0f;
                game_over = true;
                break;
            }

            tree.run_simulations(config.simulations_per_move, eval_fn);
            move_t m = tree.select_move(0.0f);
            if (m == board::nullmove) break;

            board.make_move(m);
            tree.advance(m, board);
            moves_played++;
        }

        if (!game_over) outcome = 0.0f;  // forced draw

        // Convert outcome to new model's perspective
        float new_model_outcome = new_is_white ? outcome : -outcome;

        if (new_model_outcome > 0.5f) result.new_wins++;
        else if (new_model_outcome < -0.5f) result.old_wins++;
        else result.draws++;

        const char* res_str = (new_model_outcome > 0.5f) ? "W"
                            : (new_model_outcome < -0.5f) ? "L" : "D";
        DC0_LOG_INFO("Eval game %d/%d (%s as %s): %s in %d moves | "
                "running: +%d -%d =%d",
                game + 1, num_games,
                res_str, new_is_white ? "white" : "black",
                game_over ? (outcome > 0 ? "1-0" : (outcome < 0 ? "0-1" : "draw")) : "draw",
                moves_played,
                result.new_wins, result.old_wins, result.draws);
    }

    result.win_rate = (result.new_wins + 0.5f * result.draws) / result.games;
    return result;
}

// --- Full training generation loop ---

struct GenerationConfig {
    // Self-play
    int games_per_generation = 100;
    SelfPlayConfig selfplay;

    // Training
    TrainingConfig training;

    // Evaluation
    int eval_games = 20;
    int eval_simulations = 50;
    float promotion_threshold = 0.55f;  // win rate needed to promote new model

    // Paths
    std::string output_dir = "dc0_output";

    // Loop
    int num_generations = 100;
};

struct GenerationResult {
    int generation = 0;
    SelfPlayStats selfplay_stats;
    TrainingResult training_result;
    EvalResult eval_result;
    bool model_promoted = false;
};

// Run a single generation: self-play -> train -> evaluate.
inline GenerationResult run_generation(
    int generation,
    DC0Network current_best,
    DC0Network training_model,
    torch::Device device,
    const GenerationConfig& config,
    TrainingDataset& accumulated_data
) {
    GenerationResult gen_result;
    gen_result.generation = generation;

    // Ensure output directory exists
    std::filesystem::create_directories(config.output_dir);

    // 1. Self-play with current best model
    DC0_LOG_INFO("=== Generation %d: Self-play ===", generation);
    NNEvaluator best_eval(current_best, device);
    std::string data_path = config.output_dir + "/selfplay_gen" + std::to_string(generation) + ".bin";

    gen_result.selfplay_stats = run_self_play(
        best_eval, config.games_per_generation, config.selfplay, data_path);

    // 2. Accumulate training data
    accumulated_data.append_file(data_path);

    // Trim to max window size
    // (TrainingDataset doesn't support trimming yet, but the data is there)

    // 3. Train
    DC0_LOG_INFO("=== Generation %d: Training ===", generation);

    // Copy best weights to training model
    {
        std::string tmp_path = config.output_dir + "/tmp_weights.bin";
        current_best->save_weights(tmp_path);
        training_model->load_weights(tmp_path);
        std::filesystem::remove(tmp_path);
    }

    Trainer trainer(training_model, device, config.training);
    gen_result.training_result = trainer.train(accumulated_data);

    DC0_LOG_INFO("Training done: policy_loss=%.4f value_loss=%.4f total=%.4f (%d examples)",
            gen_result.training_result.avg_policy_loss,
            gen_result.training_result.avg_value_loss,
            gen_result.training_result.avg_total_loss,
            gen_result.training_result.total_examples);

    // 4. Evaluate: new model vs current best
    DC0_LOG_INFO("=== Generation %d: Evaluation ===", generation);
    gen_result.eval_result = evaluate_models(
        training_model, current_best, device,
        config.eval_games, config.eval_simulations);

    DC0_LOG_INFO("Eval: new_wins=%d old_wins=%d draws=%d win_rate=%.2f",
            gen_result.eval_result.new_wins,
            gen_result.eval_result.old_wins,
            gen_result.eval_result.draws,
            gen_result.eval_result.win_rate);

    // 5. Promote if new model is better
    if (gen_result.eval_result.win_rate >= config.promotion_threshold) {
        DC0_LOG_INFO("New model promoted!");
        gen_result.model_promoted = true;

        // Save new best
        std::string best_path = config.output_dir + "/best_model.bin";
        training_model->save_weights(best_path);

        // Copy new weights to current best
        current_best->load_weights(best_path);
    } else {
        DC0_LOG_INFO("Keeping current best.");
        gen_result.model_promoted = false;
    }

    // Save generation checkpoint
    std::string gen_path = config.output_dir + "/model_gen" + std::to_string(generation) + ".bin";
    training_model->save_weights(gen_path);

    return gen_result;
}

} // namespace dc0
