// NNUE Test - verify compilation and basic functionality
#include <iostream>
#include <chrono>
#include <random>
#include <memory>

#include "NNUE.hpp"
#include "Board.hpp"
#include "FEN.hpp"

using namespace std::chrono;

// Benchmark forward pass
void benchmark_forward() {
    // Allocate on heap - NetworkWeights is ~21MB!
    auto weights = std::make_unique<nnue::NetworkWeights>();
    nnue::Evaluator eval;
    
    // Initialize with random weights for testing
    std::mt19937 rng(42);
    std::uniform_int_distribution<int16_t> dist16(-100, 100);
    std::uniform_int_distribution<int8_t> dist8(-50, 50);
    
    for (auto& b : weights->ft_biases) b = dist16(rng);
    for (auto& w : weights->ft_weights) w = dist16(rng);
    for (auto& b : weights->l1_biases) b = dist16(rng) * 100;
    for (auto& w : weights->l1_weights) w = dist8(rng);
    for (auto& b : weights->l2_biases) b = dist16(rng) * 100;
    for (auto& w : weights->l2_weights) w = dist8(rng);
    weights->out_bias = dist16(rng) * 100;
    for (auto& w : weights->out_weights) w = dist8(rng);
    
    eval.set_weights(weights.get());
    eval.reset();
    
    // Create test position
    Board board(fen::starting_pos);
    
    // Warmup
    for (int i = 0; i < 100; ++i) {
        eval.reset();
        eval.evaluate(board);
    }
    
    // Benchmark full evaluation (from scratch)
    constexpr int ITERATIONS = 100000;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        eval.reset();
        eval.evaluate(board);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    
    std::cout << "Full evaluation: " << (duration / ITERATIONS) << " ns/eval" << std::endl;
    std::cout << "Evaluations/sec: " << (ITERATIONS * 1000000000LL / duration) << std::endl;
    
    // Benchmark incremental update (simulated)
    eval.reset();
    eval.evaluate(board);  // Initial evaluation
    
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        eval.push_accumulator();
        // Simulate small delta (2 features changed)
        nnue::AccumulatorDelta delta;
        delta.add_removed(1000);
        delta.add_added(2000);
        eval.apply_delta(WHITE, delta);
        eval.apply_delta(BLACK, delta);
        eval.forward(WHITE);
        eval.pop_accumulator();
    }
    end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start).count();
    
    std::cout << "Incremental eval: " << (duration / ITERATIONS) << " ns/eval" << std::endl;
    std::cout << "Incremental/sec: " << (ITERATIONS * 1000000000LL / duration) << std::endl;
}

// Test feature index calculation
void test_halfkp_indices() {
    std::cout << "\n=== HalfKP Index Tests ===" << std::endl;
    
    // Test some known indices
    // White king on e1 (sq=4), white pawn on e2 (sq=12)
    // From white's perspective
    size_t idx = nnue::HalfKP::index(true, 4, 12, 0);  // piece_type 0 = white pawn from white's view
    std::cout << "White Ke1, white Pe2 (white POV): " << idx << std::endl;
    
    // Same position from black's perspective
    // Black king on e8 (sq=60), same white pawn e2 (sq=12)
    // Oriented: king at 63-60=3, pawn at 63-12=51
    idx = nnue::HalfKP::index(false, nnue::HalfKP::orient(false, 60), 12, 1);  // piece_type 1 = white pawn from black's view (opponent's pawn)
    std::cout << "Black Ke8, white Pe2 (black POV): " << idx << std::endl;
    
    std::cout << "HALFKP_SIZE = " << nnue::HALFKP_SIZE << std::endl;
}

// Test with actual board position
void test_with_board() {
    std::cout << "\n=== Board Position Test ===" << std::endl;
    
    Board board(fen::starting_pos);
    std::cout << "Starting position loaded" << std::endl;
    
    // Allocate on heap - NetworkWeights is ~21MB!
    auto weights = std::make_unique<nnue::NetworkWeights>();
    nnue::Evaluator eval;
    
    // Zero weights (output should be ~0)
    std::memset(weights.get(), 0, sizeof(nnue::NetworkWeights));
    eval.set_weights(weights.get());
    eval.reset();
    
    int32_t score = eval.evaluate(board);
    std::cout << "Score with zero weights: " << score << " cp" << std::endl;
    
    // Set all biases to 1, should get a small positive score
    for (auto& b : weights->ft_biases) b = 1;
    eval.reset();
    score = eval.evaluate(board);
    std::cout << "Score with unit biases: " << score << " cp" << std::endl;
}

// Test loading network from file
void test_network_load(const char* path) {
    std::cout << "\n=== Network Load Test ===" << std::endl;
    
    auto weights = std::make_unique<nnue::NetworkWeights>();
    
    if (!nnue::NetworkLoader::load(path, *weights)) {
        std::cout << "FAILED: Could not load " << path << std::endl;
        return;
    }
    std::cout << "Loaded: " << path << std::endl;
    
    // Create evaluator and test
    nnue::Evaluator eval;
    eval.set_weights(weights.get());
    eval.reset();
    
    Board board(fen::starting_pos);
    int32_t score = eval.evaluate(board);
    std::cout << "Starting position score: " << score << " cp" << std::endl;
    
    // Test a few other positions
    Board sicilian(fen::load_from_string("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"));
    eval.reset();
    int32_t sic_score = eval.evaluate(sicilian);
    std::cout << "Sicilian (1.e4 c5) score: " << sic_score << " cp" << std::endl;
    
    Board italian(fen::load_from_string("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"));
    eval.reset();
    int32_t ita_score = eval.evaluate(italian);
    std::cout << "Italian Game score: " << ita_score << " cp" << std::endl;
}

// Report SIMD capabilities
void report_simd() {
    std::cout << "=== SIMD Capabilities ===" << std::endl;
#if defined(NNUE_USE_AVX512)
    std::cout << "Using: AVX-512" << std::endl;
#elif defined(NNUE_USE_AVX2)
    std::cout << "Using: AVX2" << std::endl;
#elif defined(NNUE_USE_SSE42)
    std::cout << "Using: SSE4.2" << std::endl;
#elif defined(NNUE_USE_NEON)
    std::cout << "Using: ARM NEON" << std::endl;
#else
    std::cout << "Using: Scalar (no SIMD)" << std::endl;
#endif
    std::cout << "SIMD width: " << nnue::SIMD_WIDTH << " bytes" << std::endl;
}

int main(int argc, char** argv) {
    report_simd();
    test_halfkp_indices();
    test_with_board();
    
    if (argc > 1) {
        test_network_load(argv[1]);
    }
    
    benchmark_forward();
    return 0;
}
