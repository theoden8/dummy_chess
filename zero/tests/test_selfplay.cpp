// Tests for SelfPlay.hpp: serialization, random game, NN evaluator, NN game, data roundtrip.

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <filesystem>

#include <Board.hpp>
#include <FEN.hpp>

#include <SelfPlay.hpp>

namespace {

std::pair<std::vector<float>, float> random_eval(const Board& board) {
    static std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> policy(dc0::POLICY_SIZE, 0.0f);
    return {policy, dist(rng)};
}

} // namespace

// --- Tests without GPU ---

TEST(SelfPlay, TrainingDataSerialization) {
    dc0::TrainingData data;

    for (int i = 0; i < 10; ++i) {
        dc0::TrainingExample ex;
        std::memset(&ex, 0, sizeof(ex));
        ex.planes[0] = float(i);
        ex.policy[0] = float(i) * 0.1f;
        ex.result = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? -1.0f : 0.0f;
        data.add(ex);
    }

    ASSERT_EQ(data.size(), 10u);

    std::string path = "/tmp/dc0_gtest_training_data.bin";
    ASSERT_TRUE(data.save(path));

    dc0::TrainingData loaded;
    ASSERT_TRUE(loaded.load(path));
    ASSERT_EQ(loaded.size(), 10u);

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(loaded.examples[i].planes[0], float(i));
        EXPECT_NEAR(loaded.examples[i].policy[0], float(i) * 0.1f, 1e-6f);
        float expected_result = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? -1.0f : 0.0f;
        EXPECT_EQ(loaded.examples[i].result, expected_result);
    }

    std::filesystem::remove(path);
}

TEST(SelfPlay, TrainingDataBadFile) {
    dc0::TrainingData data;
    EXPECT_FALSE(data.load("/tmp/dc0_nonexistent_file_12345.bin"));
}

TEST(SelfPlay, DefaultConfig) {
    dc0::SelfPlayConfig config;
    EXPECT_EQ(config.simulations_per_move, 800);
    EXPECT_EQ(config.c_puct, 2.5f);
    EXPECT_EQ(config.max_game_moves, 512);
    EXPECT_EQ(config.temperature_moves, 30);
}

TEST(SelfPlay, PlaySingleGameRandom) {
    dc0::SelfPlayConfig config;
    config.simulations_per_move = 20;
    config.max_game_moves = 100;
    config.temperature_moves = 10;

    dc0::TrainingData data;
    dc0::GameResult result = dc0::play_self_play_game(random_eval, config, data);

    EXPECT_GT(result.num_moves, 0);
    EXPECT_TRUE(result.is_checkmate || result.is_draw);
    EXPECT_GT(data.size(), 0u);
    EXPECT_EQ(data.size(), static_cast<size_t>(result.num_moves));

    for (auto& ex : data.examples) {
        EXPECT_GE(ex.result, -1.0f);
        EXPECT_LE(ex.result, 1.0f);

        float sum = 0.0f;
        for (int i = 0; i < dc0::POLICY_SIZE; ++i) {
            sum += ex.policy[i];
        }
        EXPECT_TRUE(std::abs(sum - 1.0f) < 1e-3f || sum == 0.0f)
            << "policy sum=" << sum;
    }
}

// --- Tests with GPU + libtorch ---

TEST(SelfPlay, NNEvaluator) {
    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;

    dc0::NNEvaluator evaluator(model, device);

    fen::FEN f = fen::load_from_string(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Board board(f);

    auto [policy, value] = evaluator.evaluate(board);

    EXPECT_EQ(policy.size(), static_cast<size_t>(dc0::POLICY_SIZE));
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
}

TEST(SelfPlay, NNSelfPlayGame) {
    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    dc0::NNEvaluator evaluator(model, device);

    dc0::SelfPlayConfig config;
    config.simulations_per_move = 10;
    config.max_game_moves = 30;
    config.temperature_moves = 5;

    dc0::TrainingData data;
    auto eval_fn = evaluator.get_eval_function();
    dc0::GameResult result = dc0::play_self_play_game(eval_fn, config, data);

    EXPECT_GT(result.num_moves, 0);
    EXPECT_GT(data.size(), 0u);
}

TEST(SelfPlay, TrainingDataSaveLoadWithNN) {
    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    dc0::NNEvaluator evaluator(model, device);

    dc0::SelfPlayConfig config;
    config.simulations_per_move = 10;
    config.max_game_moves = 20;
    config.temperature_moves = 5;

    dc0::TrainingData data;
    auto eval_fn = evaluator.get_eval_function();
    dc0::play_self_play_game(eval_fn, config, data);

    std::string path = "/tmp/dc0_gtest_nn_data.bin";
    ASSERT_TRUE(data.save(path));

    dc0::TrainingData loaded;
    ASSERT_TRUE(loaded.load(path));
    ASSERT_EQ(loaded.size(), data.size());

    if (loaded.size() > 0 && data.size() > 0) {
        EXPECT_EQ(std::memcmp(loaded.examples[0].planes, data.examples[0].planes,
                              sizeof(data.examples[0].planes)), 0);
        EXPECT_EQ(loaded.examples[0].result, data.examples[0].result);
    }

    std::filesystem::remove(path);
}

TEST(SelfPlay, BatchedNNSelfPlayGame) {
    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    dc0::NNEvaluator evaluator(model, device);

    dc0::SelfPlayConfig config;
    config.simulations_per_move = 10;
    config.max_game_moves = 30;
    config.temperature_moves = 5;
    config.batch_size = 4;

    dc0::TrainingData data;
    auto batch_eval_fn = evaluator.get_batched_eval_function();
    dc0::GameResult result = dc0::play_self_play_game_batched(batch_eval_fn, config, data);

    EXPECT_GT(result.num_moves, 0);
    EXPECT_GT(data.size(), 0u);
    EXPECT_EQ(data.size(), static_cast<size_t>(result.num_moves));

    for (auto& ex : data.examples) {
        EXPECT_GE(ex.result, -1.0f);
        EXPECT_LE(ex.result, 1.0f);

        float sum = 0.0f;
        for (int i = 0; i < dc0::POLICY_SIZE; ++i) {
            sum += ex.policy[i];
        }
        EXPECT_TRUE(std::abs(sum - 1.0f) < 1e-3f || sum == 0.0f)
            << "policy sum=" << sum;
    }
}

TEST(SelfPlay, ParallelSelfPlay) {
    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    dc0::NNEvaluator evaluator(model, device);

    dc0::SelfPlayConfig config;
    config.simulations_per_move = 10;
    config.max_game_moves = 30;
    config.temperature_moves = 5;
    config.batch_size = 8;

    int num_games = 4;
    int n_parallel = 2;
    std::string output_path = "/tmp/dc0_gtest_parallel.bin";

    dc0::SelfPlayStats stats = dc0::run_self_play_parallel(
        evaluator, num_games, config, output_path, n_parallel);

    // All games should have completed
    EXPECT_EQ(stats.total_games, num_games);
    EXPECT_GT(stats.total_moves, 0);
    EXPECT_GT(stats.total_positions, 0);
    EXPECT_GT(stats.elapsed_sec, 0.0);

    // Win/draw/loss should sum to total games
    EXPECT_EQ(stats.white_wins + stats.black_wins + stats.draws, num_games);

    // Output file should be loadable
    dc0::TrainingData loaded;
    ASSERT_TRUE(loaded.load(output_path));
    EXPECT_EQ(static_cast<int>(loaded.size()), stats.total_positions);

    // Validate training examples
    for (auto& ex : loaded.examples) {
        EXPECT_GE(ex.result, -1.0f);
        EXPECT_LE(ex.result, 1.0f);

        float sum = 0.0f;
        for (int i = 0; i < dc0::POLICY_SIZE; ++i) {
            sum += ex.policy[i];
        }
        EXPECT_TRUE(std::abs(sum - 1.0f) < 1e-3f || sum == 0.0f)
            << "policy sum=" << sum;
    }

    std::filesystem::remove(output_path);
}
