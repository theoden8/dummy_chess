// Tests for Training.hpp: dataset, training step, loss decrease, checkpoint, LR schedule, train+eval.

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <filesystem>

#include <Board.hpp>
#include <FEN.hpp>

#include <Training.hpp>

namespace {

std::pair<std::vector<float>, float> random_eval(const Board& board) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> policy(dc0::POLICY_SIZE, 0.0f);
    return {policy, dist(rng)};
}

} // namespace

TEST(Training, Dataset) {
    dc0::TrainingData data;
    for (int i = 0; i < 100; ++i) {
        dc0::TrainingExample ex;
        std::memset(&ex, 0, sizeof(ex));
        ex.planes[0] = float(i) / 100.0f;
        ex.policy[i % dc0::POLICY_SIZE] = 1.0f;
        ex.result = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? -1.0f : 0.0f;
        data.add(ex);
    }

    dc0::TrainingDataset dataset(data);
    EXPECT_EQ(dataset.num_examples(), 100u);

    auto example = dataset.get(0);
    EXPECT_EQ(example.data.sizes(), (torch::IntArrayRef{dc0::ENCODING_PLANES, 8, 8}));
    EXPECT_EQ(example.target.sizes(), (torch::IntArrayRef{dc0::POLICY_SIZE + 1}));
}

TEST(Training, TrainingStep) {
    dc0::SelfPlayConfig sp_config;
    sp_config.simulations_per_move = 10;
    sp_config.max_game_moves = 30;

    dc0::TrainingData data;
    for (int g = 0; g < 3; ++g) {
        dc0::play_self_play_game(random_eval, sp_config, data);
    }

    ASSERT_GT(data.size(), 0u);

    dc0::TrainingDataset dataset(data);

    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;

    dc0::TrainingConfig config;
    config.batch_size = 16;
    config.epochs = 1;
    config.log_interval = 0;

    dc0::Trainer trainer(model, device, config);
    dc0::TrainingResult result = trainer.train(dataset);

    EXPECT_GT(result.total_batches, 0);
    EXPECT_GT(result.avg_policy_loss, 0.0f);
    EXPECT_GT(result.avg_value_loss, 0.0f);
    EXPECT_GT(result.avg_total_loss, 0.0f);
    EXPECT_TRUE(std::isfinite(result.avg_total_loss));
}

TEST(Training, LossDecreases) {
    dc0::SelfPlayConfig sp_config;
    sp_config.simulations_per_move = 10;
    sp_config.max_game_moves = 30;

    dc0::TrainingData data;
    for (int g = 0; g < 5; ++g) {
        dc0::play_self_play_game(random_eval, sp_config, data);
    }
    dc0::TrainingDataset dataset(data);

    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;

    dc0::TrainingConfig config;
    config.batch_size = 16;
    config.epochs = 1;
    config.log_interval = 0;
    config.learning_rate = 0.01f;

    dc0::Trainer trainer(model, device, config);

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int pass = 0; pass < 5; ++pass) {
        dc0::TrainingResult result = trainer.train(dataset);
        if (pass == 0) first_loss = result.avg_total_loss;
        last_loss = result.avg_total_loss;
    }

    EXPECT_LT(last_loss, first_loss) << "loss should decrease over training";
}

TEST(Training, CheckpointSaveLoad) {
    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;

    dc0::TrainingConfig config;
    dc0::Trainer trainer(model, device, config);

    std::string path = "/tmp/dc0_gtest_checkpoint.bin";
    trainer.save_checkpoint(path);
    ASSERT_TRUE(std::filesystem::exists(path));

    // Compare on CPU to avoid cross-device issues
    model->to(torch::kCPU);
    model->eval();
    auto input = torch::randn({1, dc0::ENCODING_PLANES, 8, 8});
    torch::NoGradGuard no_grad;
    auto [p1, v1] = model->forward(input);

    dc0::DC0Network model2(/*n_blocks=*/2, /*n_filters=*/32);
    model2->load_weights(path);
    model2->eval();

    auto [p2, v2] = model2->forward(input);

    float policy_diff = (p1 - p2).abs().max().item<float>();
    float value_diff = (v1 - v2).abs().max().item<float>();
    EXPECT_LT(policy_diff, 1e-5f);
    EXPECT_LT(value_diff, 1e-5f);

    std::filesystem::remove(path);
}

TEST(Training, LRSchedule) {
    dc0::TrainingConfig config;
    config.learning_rate = 0.1f;
    config.lr_steps = {10, 20};
    config.lr_gamma = 0.1f;

    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::kCPU;

    dc0::Trainer trainer(model, device, config);
    EXPECT_NEAR(trainer.current_lr(), 0.1f, 1e-6f);
}

TEST(Training, NNEvaluatorInTraining) {
    dc0::DC0Network model(/*n_blocks=*/2, /*n_filters=*/32);
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0) : torch::kCPU;

    dc0::SelfPlayConfig sp_config;
    sp_config.simulations_per_move = 10;
    sp_config.max_game_moves = 20;

    dc0::TrainingData data;
    dc0::play_self_play_game(random_eval, sp_config, data);
    dc0::TrainingDataset dataset(data);

    dc0::TrainingConfig config;
    config.batch_size = 16;
    config.epochs = 1;
    config.log_interval = 0;

    dc0::Trainer trainer(model, device, config);
    trainer.train(dataset);

    dc0::NNEvaluator evaluator(model, device);
    auto eval_fn = evaluator.get_eval_function();

    dc0::TrainingData new_data;
    dc0::GameResult result = dc0::play_self_play_game(eval_fn, sp_config, new_data);

    EXPECT_GT(result.num_moves, 0);
    EXPECT_GT(new_data.size(), 0u);
}
