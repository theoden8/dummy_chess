// Tests for MCTS.hpp: search, selection, policy, terminal, noise, tree reuse, game play.

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>

#include <Board.hpp>
#include <FEN.hpp>

#include <MCTS.hpp>

namespace {

std::pair<std::vector<float>, float> random_eval(const Board& board) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> policy(dc0::POLICY_SIZE, 0.0f);
    return {policy, dist(rng)};
}

// Batched random eval for testing run_simulations_batched
void random_batch_eval(
    const float* planes, const bool* masks, int n,
    float* out_policies, float* out_values
) {
    static std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::memset(out_policies, 0, n * dc0::POLICY_SIZE * sizeof(float));
    for (int i = 0; i < n; ++i) {
        out_values[i] = dist(rng);
    }
}

Board starting_board() {
    fen::FEN f = fen::load_from_string(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    return Board(f);
}

} // namespace

TEST(MCTS, BasicSearch) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.set_root(board);
    tree.run_simulations(100, random_eval);

    EXPECT_GT(tree.root_visits(), 0u);

    auto stats = tree.get_move_stats();
    EXPECT_FALSE(stats.empty());
    EXPECT_EQ(stats.size(), 20u);

    move_t m = tree.select_move(0.0f);
    EXPECT_NE(m, board::nullmove);
}

TEST(MCTS, DeterministicSelect) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.set_root(board);
    tree.run_simulations(200, random_eval);

    move_t m1 = tree.select_move(0.0f);
    move_t m2 = tree.select_move(0.0f);
    EXPECT_EQ(m1, m2);
}

TEST(MCTS, PolicyOutput) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.set_root(board);
    tree.run_simulations(100, random_eval);

    float policy[dc0::POLICY_SIZE];
    tree.get_policy(policy, 1.0f);

    float sum = 0.0f;
    int nonzero = 0;
    for (int i = 0; i < dc0::POLICY_SIZE; ++i) {
        sum += policy[i];
        if (policy[i] > 0.0f) ++nonzero;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
    EXPECT_EQ(nonzero, 20);
}

TEST(MCTS, TerminalCheckmate) {
    fen::FEN f = fen::load_from_string(
        "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    Board board(f);

    ASSERT_TRUE(board.is_checkmate());

    dc0::MCTSTree tree;
    tree.set_root(board);
    tree.run_simulations(10, random_eval);

    EXPECT_TRUE(tree.is_terminal());
    EXPECT_TRUE(tree.root()->edges.empty());
}

TEST(MCTS, DirichletNoise) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.add_noise = true;
    tree.set_root(board);
    tree.run_simulations(50, random_eval);

    auto stats = tree.get_move_stats();
    float prior_sum = 0.0f;
    for (auto& s : stats) {
        prior_sum += s.prior;
    }
    EXPECT_NEAR(prior_sum, 1.0f, 1e-3f);
}

TEST(MCTS, TreeAdvance) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.set_root(board);
    tree.run_simulations(200, random_eval);

    move_t best = tree.select_move(0.0f);
    uint32_t visits_before = tree.root_visits();

    Board new_board(board);
    new_board.make_move(best);
    tree.advance(best, new_board);

    uint32_t visits_after = tree.root_visits();
    EXPECT_GT(visits_after, 0u);
    EXPECT_LT(visits_after, visits_before);
}

TEST(MCTS, PlaysLegalGame) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.set_root(board);

    int moves_played = 0;
    const int max_moves = 200;

    while (moves_played < max_moves) {
        if (board.is_checkmate() || board.is_draw()) break;

        tree.run_simulations(50, random_eval);
        move_t m = tree.select_move(1.0f);
        if (m == board::nullmove) break;

        board.make_move(m);
        tree.advance(m, board);
        ++moves_played;
    }

    EXPECT_GT(moves_played, 0);
    EXPECT_GE(moves_played, 10);
}

TEST(MCTS, BatchedBasicSearch) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.set_root(board);
    tree.run_simulations_batched(100, random_batch_eval, /*batch_size=*/8);

    EXPECT_GT(tree.root_visits(), 0u);

    auto stats = tree.get_move_stats();
    EXPECT_FALSE(stats.empty());
    EXPECT_EQ(stats.size(), 20u);

    move_t m = tree.select_move(0.0f);
    EXPECT_NE(m, board::nullmove);
}

TEST(MCTS, BatchedPolicyOutput) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.set_root(board);
    tree.run_simulations_batched(100, random_batch_eval, /*batch_size=*/16);

    float policy[dc0::POLICY_SIZE];
    tree.get_policy(policy, 1.0f);

    float sum = 0.0f;
    int nonzero = 0;
    for (int i = 0; i < dc0::POLICY_SIZE; ++i) {
        sum += policy[i];
        if (policy[i] > 0.0f) ++nonzero;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
    EXPECT_EQ(nonzero, 20);
}

TEST(MCTS, BatchedPlaysLegalGame) {
    Board board = starting_board();

    dc0::MCTSTree tree;
    tree.set_root(board);

    int moves_played = 0;
    const int max_moves = 200;

    while (moves_played < max_moves) {
        if (board.is_checkmate() || board.is_draw()) break;

        tree.run_simulations_batched(50, random_batch_eval, /*batch_size=*/8);
        move_t m = tree.select_move(1.0f);
        if (m == board::nullmove) break;

        board.make_move(m);
        tree.advance(m, board);
        ++moves_played;
    }

    EXPECT_GT(moves_played, 0);
    EXPECT_GE(moves_played, 10);
}

TEST(MCTS, BatchedTerminalCheckmate) {
    fen::FEN f = fen::load_from_string(
        "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    Board board(f);

    ASSERT_TRUE(board.is_checkmate());

    dc0::MCTSTree tree;
    tree.set_root(board);
    tree.run_simulations_batched(10, random_batch_eval, /*batch_size=*/4);

    EXPECT_TRUE(tree.is_terminal());
    EXPECT_TRUE(tree.root()->edges.empty());
}
