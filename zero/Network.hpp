#pragma once

// dc0 neural network: SE-ResNet with policy + WDL value heads.
//
// Architecture follows LC0/AlphaZero with Squeeze-Excitation residual blocks.
// Defined using libtorch (torch::nn) so it supports both training and inference.
//
// Input:  (batch, 22, 8, 8) float tensor
// Output: policy logits (batch, 4672), WDL logits (batch, 3)

#include <torch/torch.h>

#include <MoveEncoding.hpp>

namespace dc0 {

static constexpr int NUM_INPUT_PLANES = 22;

// --- Squeeze-Excitation block ---

struct SEBlockImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    int channels_;

    SEBlockImpl(int channels, int se_ratio = 4)
        : channels_(channels) {
        int mid = channels / se_ratio;
        fc1 = register_module("fc1", torch::nn::Linear(channels, mid));
        fc2 = register_module("fc2", torch::nn::Linear(mid, 2 * channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: (batch, channels, 8, 8)
        auto b = x.size(0);
        auto c = x.size(1);

        // Global average pooling
        auto s = x.mean({2, 3});  // (batch, channels)

        // FC -> ReLU -> FC
        s = torch::relu(fc1->forward(s));
        s = fc2->forward(s);  // (batch, 2*channels)

        // Split into scale (sigmoid) and bias
        auto wb = s.view({b, 2, c, 1, 1});
        auto w = torch::sigmoid(wb.select(1, 0));   // (batch, channels, 1, 1)
        auto bias = wb.select(1, 1);                 // (batch, channels, 1, 1)

        return w * x + bias;
    }
};
TORCH_MODULE(SEBlock);

// --- SE-Residual block ---

struct ResBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    SEBlock se{nullptr};

    ResBlockImpl(int channels, int se_ratio = 4) {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
        se = register_module("se", SEBlock(channels, se_ratio));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        auto out = torch::relu(bn1->forward(conv1->forward(x)));
        out = bn2->forward(conv2->forward(out));
        out = se->forward(out);
        return torch::relu(out + residual);
    }
};
TORCH_MODULE(ResBlock);

// --- Main network ---

struct DC0NetworkImpl : torch::nn::Module {
    // Config (stored for serialization)
    int n_blocks_, n_filters_, se_ratio_;
    int policy_channels_, value_channels_, value_fc_size_;

    // Input block
    torch::nn::Conv2d input_conv{nullptr};
    torch::nn::BatchNorm2d input_bn{nullptr};

    // Residual tower
    torch::nn::ModuleList blocks{nullptr};

    // Policy head
    torch::nn::Conv2d policy_conv1{nullptr}, policy_conv2{nullptr};
    torch::nn::BatchNorm2d policy_bn{nullptr};

    // Value head
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::BatchNorm2d value_bn{nullptr};
    torch::nn::Linear value_fc1{nullptr}, value_fc2{nullptr};

    DC0NetworkImpl(
        int n_blocks = 6,
        int n_filters = 128,
        int se_ratio = 4,
        int policy_channels = 32,
        int value_channels = 32,
        int value_fc_size = 128
    ) : n_blocks_(n_blocks), n_filters_(n_filters), se_ratio_(se_ratio),
        policy_channels_(policy_channels), value_channels_(value_channels),
        value_fc_size_(value_fc_size)
    {
        // Input conv
        input_conv = register_module("input_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_INPUT_PLANES, n_filters, 3).padding(1).bias(false)));
        input_bn = register_module("input_bn", torch::nn::BatchNorm2d(n_filters));

        // Residual tower
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < n_blocks; ++i) {
            blocks->push_back(ResBlock(n_filters, se_ratio));
        }

        // Policy head
        policy_conv1 = register_module("policy_conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(n_filters, policy_channels, 1).bias(false)));
        policy_bn = register_module("policy_bn", torch::nn::BatchNorm2d(policy_channels));
        policy_conv2 = register_module("policy_conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(policy_channels, NUM_MOVE_TYPES, 1)));

        // Value head
        value_conv = register_module("value_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(n_filters, value_channels, 1).bias(false)));
        value_bn = register_module("value_bn", torch::nn::BatchNorm2d(value_channels));
        value_fc1 = register_module("value_fc1",
            torch::nn::Linear(value_channels * 64, value_fc_size));
        value_fc2 = register_module("value_fc2",
            torch::nn::Linear(value_fc_size, 3));
    }

    // Forward pass: returns {policy_logits (batch, 4672), wdl_logits (batch, 3)}
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // Input block
        auto out = torch::relu(input_bn->forward(input_conv->forward(x)));

        // Residual tower
        for (auto& module : *blocks) {
            out = module->as<ResBlockImpl>()->forward(out);
        }

        // Policy head: conv -> BN -> ReLU -> conv -> permute -> flatten
        auto p = torch::relu(policy_bn->forward(policy_conv1->forward(out)));
        p = policy_conv2->forward(p);  // (batch, 73, 8, 8)
        // Permute to (batch, 8, 8, 73) = (batch, rank, file, move_type)
        // Then flatten to (batch, 4672) where index = (rank*8 + file)*73 + mt
        p = p.permute({0, 2, 3, 1}).contiguous();
        p = p.view({p.size(0), POLICY_SIZE});

        // Value head: conv -> BN -> ReLU -> flatten -> FC -> ReLU -> FC
        auto v = torch::relu(value_bn->forward(value_conv->forward(out)));
        v = v.view({v.size(0), -1});
        v = torch::relu(value_fc1->forward(v));
        v = value_fc2->forward(v);  // (batch, 3)

        return {p, v};
    }

    // Forward with legal move masking and softmax
    std::pair<torch::Tensor, torch::Tensor> predict(
        torch::Tensor x, torch::Tensor legal_mask
    ) {
        auto [policy_logits, wdl_logits] = forward(x);

        // Mask illegal moves with -inf before softmax
        policy_logits = policy_logits.masked_fill(~legal_mask, -1e32f);
        auto policy = torch::softmax(policy_logits, /*dim=*/1);
        auto wdl = torch::softmax(wdl_logits, /*dim=*/1);

        return {policy, wdl};
    }

    // Convert WDL probabilities to scalar value in [-1, 1]
    // wdl: (batch, 3) = [P(win), P(draw), P(loss)]
    static torch::Tensor value_from_wdl(torch::Tensor wdl) {
        return wdl.select(1, 0) - wdl.select(1, 2);
    }

    // Save model weights
    void save_weights(const std::string& path) {
        torch::save({}, path);  // placeholder — use serialize below
        // Actually save all parameters
        auto params = named_parameters();
        auto buffers = named_buffers();
        std::vector<torch::Tensor> tensors;
        std::vector<std::string> keys;
        for (auto& p : params) {
            keys.push_back(p.key());
            tensors.push_back(p.value());
        }
        for (auto& b : buffers) {
            keys.push_back(b.key());
            tensors.push_back(b.value());
        }
        torch::serialize::OutputArchive archive;
        for (size_t i = 0; i < keys.size(); ++i) {
            archive.write(keys[i], tensors[i]);
        }
        archive.save_to(path);
    }

    // Load model weights
    void load_weights(const std::string& path) {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        auto params = named_parameters();
        auto buffers = named_buffers();
        for (auto& p : params) {
            torch::Tensor t;
            if (archive.try_read(p.key(), t)) {
                p.value().data().copy_(t);
            }
        }
        for (auto& b : buffers) {
            torch::Tensor t;
            if (archive.try_read(b.key(), t)) {
                b.value().data().copy_(t);
            }
        }
    }
};
TORCH_MODULE(DC0Network);

} // namespace dc0
