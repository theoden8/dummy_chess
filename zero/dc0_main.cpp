// dc0: AlphaZero-style chess engine — main CLI binary.
//
// Modes:
//   dc0 train [options]      — Run the full self-play + training loop
//   dc0 selfplay [options]   — Generate self-play data only
//   dc0 eval [options]       — Evaluate model in self-play games (with stats)
//
// Common options:
//   --model PATH             — Load model weights from PATH
//   --output DIR             — Output directory (default: dc0_output)
//   --device DEVICE          — "cuda:0", "cpu", etc. (default: auto-detect)
//   --blocks N               — Network blocks (default: 6)
//   --filters N              — Network filters (default: 128)
//
// Train options:
//   --generations N          — Number of generations (default: 100)
//   --games N                — Self-play games per generation (default: 100)
//   --sims N                 — MCTS simulations per move (default: 800)
//   --batch-size N           — Training batch size (default: 256)
//   --lr FLOAT               — Learning rate (default: 0.02)
//   --epochs N               — Training epochs per generation (default: 1)
//   --eval-games N           — Evaluation games per generation (default: 20)
//   --eval-sims N            — Simulations per move during eval (default: 50)
//
// Selfplay options:
//   --games N                — Number of games to play (default: 100)
//   --sims N                 — MCTS simulations per move (default: 800)
//   --output-file PATH       — Output file for training data

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <filesystem>

#include <m42.h>
#include <Zobrist.hpp>

#include <Logging.hpp>
#include <Training.hpp>

// Simple argument parser
struct Args {
    std::string mode;         // "train", "selfplay", "eval"
    std::string model_path;
    std::string output_dir = "dc0_output";
    std::string output_file;
    std::string device_str;

    // Network
    int blocks = 6;
    int filters = 128;

    // Training loop
    int generations = 100;
    int games = 100;
    int sims = 800;
    int batch_size = 256;
    float lr = 0.02f;
    int epochs = 1;
    int eval_games = 20;
    int eval_sims = 50;
    float promotion_threshold = 0.55f;

    // Self-play
    int mcts_batch_size = 64;
    int temperature_moves = 30;
    int max_game_moves = 512;
    int parallel_games = 8;

    // Logging
    dc0::log::Level log_level = dc0::log::Level::INFO;
};

static void print_usage(const char* argv0) {
    fprintf(stderr, "Usage: %s <mode> [options]\n", argv0);
    fprintf(stderr, "\nModes:\n");
    fprintf(stderr, "  train      Run the full self-play + training loop\n");
    fprintf(stderr, "  selfplay   Generate self-play data only\n");
    fprintf(stderr, "  eval       Evaluate model quality via self-play\n");
    fprintf(stderr, "\nCommon options:\n");
    fprintf(stderr, "  --model PATH         Load model weights\n");
    fprintf(stderr, "  --output DIR         Output directory (default: dc0_output)\n");
    fprintf(stderr, "  --device DEVICE      Device: cuda:0, cpu (default: auto)\n");
    fprintf(stderr, "  --blocks N           Network blocks (default: 6)\n");
    fprintf(stderr, "  --filters N          Network filters (default: 128)\n");
    fprintf(stderr, "  -v, --verbose        Debug-level logging\n");
    fprintf(stderr, "  -q, --quiet          Warnings and errors only\n");
    fprintf(stderr, "\nTrain options:\n");
    fprintf(stderr, "  --generations N      Generations (default: 100)\n");
    fprintf(stderr, "  --games N            Self-play games/generation (default: 100)\n");
    fprintf(stderr, "  --sims N             MCTS sims/move (default: 800)\n");
    fprintf(stderr, "  --batch-size N       Batch size (default: 256)\n");
    fprintf(stderr, "  --lr FLOAT           Learning rate (default: 0.02)\n");
    fprintf(stderr, "  --epochs N           Epochs/generation (default: 1)\n");
    fprintf(stderr, "  --eval-games N       Eval games (default: 20)\n");
    fprintf(stderr, "  --eval-sims N        Eval sims/move (default: 50)\n");
    fprintf(stderr, "\nSelfplay options:\n");
    fprintf(stderr, "  --games N            Number of games (default: 100)\n");
    fprintf(stderr, "  --sims N             MCTS sims/move (default: 800)\n");
    fprintf(stderr, "  --mcts-batch-size N  MCTS inference batch size (default: 64)\n");
    fprintf(stderr, "  --parallel-games N   Games in flight for parallel self-play (default: 8)\n");
    fprintf(stderr, "  --output-file PATH   Output file for training data\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s train --games 50 --sims 200 --generations 10\n", argv0);
    fprintf(stderr, "  %s selfplay --model best.bin --games 100 --sims 400\n", argv0);
    fprintf(stderr, "  %s eval --model best.bin --games 20 --sims 200\n", argv0);
}

static Args parse_args(int argc, char** argv) {
    Args args;

    if (argc < 2) {
        print_usage(argv[0]);
        std::exit(1);
    }

    args.mode = argv[1];
    if (args.mode != "train" && args.mode != "selfplay" && args.mode != "eval") {
        if (args.mode == "--help" || args.mode == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        fprintf(stderr, "Unknown mode: %s\n", args.mode.c_str());
        print_usage(argv[0]);
        std::exit(1);
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", arg.c_str());
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--model") args.model_path = next();
        else if (arg == "--output") args.output_dir = next();
        else if (arg == "--output-file") args.output_file = next();
        else if (arg == "--device") args.device_str = next();
        else if (arg == "--blocks") args.blocks = std::atoi(next());
        else if (arg == "--filters") args.filters = std::atoi(next());
        else if (arg == "--generations") args.generations = std::atoi(next());
        else if (arg == "--games") args.games = std::atoi(next());
        else if (arg == "--sims") args.sims = std::atoi(next());
        else if (arg == "--batch-size") args.batch_size = std::atoi(next());
        else if (arg == "--lr") args.lr = std::atof(next());
        else if (arg == "--epochs") args.epochs = std::atoi(next());
        else if (arg == "--eval-games") args.eval_games = std::atoi(next());
        else if (arg == "--eval-sims") args.eval_sims = std::atoi(next());
        else if (arg == "--mcts-batch-size") args.mcts_batch_size = std::atoi(next());
        else if (arg == "--parallel-games") args.parallel_games = std::atoi(next());
        else if (arg == "--temperature-moves") args.temperature_moves = std::atoi(next());
        else if (arg == "--max-game-moves") args.max_game_moves = std::atoi(next());
        else if (arg == "--promotion-threshold") args.promotion_threshold = std::atof(next());
        else if (arg == "--verbose" || arg == "-v") args.log_level = dc0::log::Level::DEBUG;
        else if (arg == "--quiet" || arg == "-q") args.log_level = dc0::log::Level::WARN;
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    return args;
}

static torch::Device get_device(const Args& args) {
    if (!args.device_str.empty()) {
        return torch::Device(args.device_str);
    }
    if (torch::cuda::is_available()) {
        DC0_LOG_INFO("Using CUDA device 0");
        return torch::Device(torch::kCUDA, 0);
    }
    DC0_LOG_WARN("CUDA not available, using CPU");
    return torch::kCPU;
}

// --- train mode ---

static int cmd_train(const Args& args) {
    torch::Device device = get_device(args);

    DC0_LOG_INFO("Network: %d blocks, %d filters", args.blocks, args.filters);
    DC0_LOG_INFO("Training: %d generations, %d games/gen, %d sims/move, %d parallel",
            args.generations, args.games, args.sims, args.parallel_games);
    DC0_LOG_INFO("          batch_size=%d, lr=%.4f, epochs=%d",
            args.batch_size, args.lr, args.epochs);
    DC0_LOG_INFO("Eval:     %d games, %d sims/move, threshold=%.2f",
            args.eval_games, args.eval_sims, args.promotion_threshold);
    DC0_LOG_INFO("Output:   %s", args.output_dir.c_str());

    // Create models
    dc0::DC0Network best_model(args.blocks, args.filters);
    dc0::DC0Network train_model(args.blocks, args.filters);

    // Load weights if provided
    if (!args.model_path.empty()) {
        DC0_LOG_INFO("Loading model from %s", args.model_path.c_str());
        best_model->to(torch::kCPU);
        best_model->load_weights(args.model_path);
        train_model->to(torch::kCPU);
        train_model->load_weights(args.model_path);
    }

    // Build generation config
    dc0::GenerationConfig gen_config;
    gen_config.games_per_generation = args.games;
    gen_config.selfplay.simulations_per_move = args.sims;
    gen_config.selfplay.temperature_moves = args.temperature_moves;
    gen_config.selfplay.max_game_moves = args.max_game_moves;
    gen_config.selfplay.batch_size = args.mcts_batch_size;
    gen_config.selfplay.n_parallel = args.parallel_games;
    gen_config.training.batch_size = args.batch_size;
    gen_config.training.learning_rate = args.lr;
    gen_config.training.epochs = args.epochs;
    gen_config.training.log_interval = 50;
    gen_config.eval_games = args.eval_games;
    gen_config.eval_simulations = args.eval_sims;
    gen_config.promotion_threshold = args.promotion_threshold;
    gen_config.output_dir = args.output_dir;
    gen_config.num_generations = args.generations;

    // Accumulated training data
    dc0::TrainingDataset accumulated_data;

    // Run generations
    for (int gen = 0; gen < args.generations; ++gen) {
        dc0::GenerationResult result = dc0::run_generation(
            gen, best_model, train_model, device, gen_config, accumulated_data);

        DC0_LOG_INFO("--- Generation %d summary ---", gen);
        DC0_LOG_INFO("  Self-play: %d games, %d positions, W/B/D = %d/%d/%d",
                result.selfplay_stats.total_games,
                result.selfplay_stats.total_positions,
                result.selfplay_stats.white_wins,
                result.selfplay_stats.black_wins,
                result.selfplay_stats.draws);
        DC0_LOG_INFO("  Training:  loss=%.4f (policy=%.4f, value=%.4f), %d examples",
                result.training_result.avg_total_loss,
                result.training_result.avg_policy_loss,
                result.training_result.avg_value_loss,
                result.training_result.total_examples);
        DC0_LOG_INFO("  Eval:      win_rate=%.2f (W=%d, L=%d, D=%d) %s",
                result.eval_result.win_rate,
                result.eval_result.new_wins,
                result.eval_result.old_wins,
                result.eval_result.draws,
                result.model_promoted ? "-> PROMOTED" : "-> kept old");
    }

    // Save final model
    std::string final_path = args.output_dir + "/final_model.bin";
    best_model->to(torch::kCPU);
    best_model->save_weights(final_path);
    DC0_LOG_INFO("Final model saved to %s", final_path.c_str());

    return 0;
}

// --- selfplay mode ---

static int cmd_selfplay(const Args& args) {
    torch::Device device = get_device(args);

    DC0_LOG_INFO("Network: %d blocks, %d filters", args.blocks, args.filters);
    DC0_LOG_INFO("Self-play: %d games, %d sims/move, %d parallel",
            args.games, args.sims, args.parallel_games);

    // Create model
    dc0::DC0Network model(args.blocks, args.filters);

    if (!args.model_path.empty()) {
        DC0_LOG_INFO("Loading model from %s", args.model_path.c_str());
        model->to(torch::kCPU);
        model->load_weights(args.model_path);
    } else {
        DC0_LOG_INFO("No model specified, using random initialization");
    }

    dc0::NNEvaluator evaluator(model, device);

    dc0::SelfPlayConfig config;
    config.simulations_per_move = args.sims;
    config.temperature_moves = args.temperature_moves;
    config.max_game_moves = args.max_game_moves;
    config.batch_size = args.mcts_batch_size;
    config.n_parallel = args.parallel_games;

    // Determine output path
    std::string output_path = args.output_file;
    if (output_path.empty()) {
        std::filesystem::create_directories(args.output_dir);
        output_path = args.output_dir + "/selfplay_data.bin";
    }

    dc0::SelfPlayStats stats;
    if (config.n_parallel > 1) {
        stats = dc0::run_self_play_parallel(
            evaluator, args.games, config, output_path, config.n_parallel);
    } else {
        stats = dc0::run_self_play(
            evaluator, args.games, config, output_path);
    }

    DC0_LOG_INFO("Self-play complete:");
    DC0_LOG_INFO("  Games: %d, Positions: %d", stats.total_games, stats.total_positions);
    DC0_LOG_INFO("  W/B/D: %d/%d/%d", stats.white_wins, stats.black_wins, stats.draws);
    DC0_LOG_INFO("  Avg moves/game: %.0f",
            stats.total_games > 0 ? double(stats.total_moves) / stats.total_games : 0.0);
    DC0_LOG_INFO("  Data saved to: %s", output_path.c_str());

    return 0;
}

// --- eval mode ---

static int cmd_eval(const Args& args) {
    torch::Device device = get_device(args);

    DC0_LOG_INFO("Network: %d blocks, %d filters", args.blocks, args.filters);
    DC0_LOG_INFO("Eval: %d games, %d sims/move", args.games, args.sims);

    // Create model
    dc0::DC0Network model(args.blocks, args.filters);

    if (!args.model_path.empty()) {
        DC0_LOG_INFO("Loading model from %s", args.model_path.c_str());
        model->to(torch::kCPU);
        model->load_weights(args.model_path);
    } else {
        DC0_LOG_INFO("No model specified, using random initialization");
    }

    dc0::NNEvaluator evaluator(model, device);

    dc0::SelfPlayConfig config;
    config.simulations_per_move = args.sims;
    config.temperature_moves = 0;  // deterministic for eval
    config.temperature = 0.0f;
    config.max_game_moves = args.max_game_moves;

    auto eval_fn = evaluator.get_eval_function();

    int white_wins = 0, black_wins = 0, draws = 0;
    int total_moves = 0;

    for (int game = 0; game < args.games; ++game) {
        dc0::TrainingData dummy;
        dc0::GameResult result = dc0::play_self_play_game(eval_fn, config, dummy);

        total_moves += result.num_moves;
        if (result.outcome > 0.5f) white_wins++;
        else if (result.outcome < -0.5f) black_wins++;
        else draws++;

        if ((game + 1) % 5 == 0 || game == args.games - 1) {
            DC0_LOG_INFO("Game %d/%d: %d moves, outcome=%.0f (running: W=%d B=%d D=%d)",
                    game + 1, args.games, result.num_moves, result.outcome,
                    white_wins, black_wins, draws);
        }
    }

    DC0_LOG_INFO("Eval complete:");
    DC0_LOG_INFO("  Games: %d", args.games);
    DC0_LOG_INFO("  W/B/D: %d/%d/%d", white_wins, black_wins, draws);
    DC0_LOG_INFO("  Avg moves/game: %.0f",
            args.games > 0 ? double(total_moves) / args.games : 0.0);

    return 0;
}

int main(int argc, char** argv) {
    // Init engine internals
    M42::init();
    zobrist::init(1 << 16);

    Args args = parse_args(argc, argv);

    // Init logging
    dc0::log::reset_clock();
    dc0::log::set_level(args.log_level);

    if (args.mode == "train") return cmd_train(args);
    if (args.mode == "selfplay") return cmd_selfplay(args);
    if (args.mode == "eval") return cmd_eval(args);

    fprintf(stderr, "Unknown mode: %s\n", args.mode.c_str());
    return 1;
}
