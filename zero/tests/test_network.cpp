// Tests for Network.hpp: shapes, masking, param count, save/load, GPU, WDL conversion.

#include <gtest/gtest.h>
#include <cmath>

#include <torch/torch.h>

#include <Network.hpp>
#include <MoveEncoding.hpp>

class NetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        net = dc0::DC0Network(6, 128, 4, 32, 32, 128);
    }

    dc0::DC0Network net{nullptr};
};

TEST_F(NetworkTest, ForwardShapes) {
    auto x = torch::randn({4, dc0::NUM_INPUT_PLANES, 8, 8});
    auto [policy, wdl] = net->forward(x);

    EXPECT_EQ(policy.sizes(), (torch::IntArrayRef{4, dc0::POLICY_SIZE}));
    EXPECT_EQ(wdl.sizes(), (torch::IntArrayRef{4, 3}));
}

TEST_F(NetworkTest, PredictMasking) {
    auto x = torch::randn({4, dc0::NUM_INPUT_PLANES, 8, 8});
    auto mask = torch::zeros({4, dc0::POLICY_SIZE}, torch::kBool);
    mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 20)}, true);

    auto [policy, wdl] = net->predict(x, mask);

    // Policy should sum to ~1 per batch
    auto policy_sum = policy.sum(1);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(policy_sum[i].item<float>(), 1.0f, 1e-4f);
    }

    // WDL should sum to ~1 per batch
    auto wdl_sum = wdl.sum(1);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(wdl_sum[i].item<float>(), 1.0f, 1e-4f);
    }

    // Illegal moves should have 0 probability
    auto illegal_sum = policy.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(20, dc0::POLICY_SIZE)}).sum().item<float>();
    EXPECT_LT(illegal_sum, 1e-6f);
}

TEST_F(NetworkTest, ParameterCount) {
    int64_t total = 0;
    for (auto& p : net->parameters()) {
        total += p.numel();
    }
    EXPECT_EQ(total, 2146988);
}

TEST_F(NetworkTest, SaveLoadRoundtrip) {
    auto x = torch::ones({1, dc0::NUM_INPUT_PLANES, 8, 8});
    auto [p1, v1] = net->forward(x);

    const char* path = "/tmp/dc0_gtest_weights.pt";
    net->save_weights(path);

    dc0::DC0Network net2(6, 128, 4, 32, 32, 128);
    net2->load_weights(path);

    auto [p2, v2] = net2->forward(x);

    float policy_diff = (p1 - p2).abs().max().item<float>();
    float wdl_diff = (v1 - v2).abs().max().item<float>();
    EXPECT_LT(policy_diff, 1e-5f);
    EXPECT_LT(wdl_diff, 1e-5f);

    std::remove(path);
}

TEST_F(NetworkTest, GPU) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto device = torch::Device(torch::kCUDA, 0);
    net->to(device);

    auto x = torch::randn({8, dc0::NUM_INPUT_PLANES, 8, 8}, device);
    auto mask = torch::zeros({8, dc0::POLICY_SIZE},
        torch::TensorOptions().dtype(torch::kBool).device(device));
    mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 20)}, true);

    auto [policy, wdl] = net->predict(x, mask);

    EXPECT_TRUE(policy.device().is_cuda());
    EXPECT_TRUE(wdl.device().is_cuda());
    EXPECT_EQ(policy.sizes(), (torch::IntArrayRef{8, dc0::POLICY_SIZE}));
}

TEST_F(NetworkTest, ValueFromWDL) {
    auto wdl = torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}});
    auto v = dc0::DC0NetworkImpl::value_from_wdl(wdl);
    EXPECT_NEAR(v[0].item<float>(), 1.0f, 1e-6f);
    EXPECT_NEAR(v[1].item<float>(), -1.0f, 1e-6f);
    EXPECT_NEAR(v[2].item<float>(), 0.0f, 1e-6f);
}
