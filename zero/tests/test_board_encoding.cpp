// Tests for BoardEncoding.hpp: plane values, legal mask, promotions, castling, halfmove.

#include <gtest/gtest.h>

#include <Board.hpp>
#include <FEN.hpp>

#include <BoardEncoding.hpp>

class BoardEncodingTest : public ::testing::Test {
protected:
    float planes[dc0::ENCODING_SIZE];
};

TEST_F(BoardEncodingTest, EncodingSize) {
    EXPECT_EQ(dc0::ENCODING_PLANES, 22);
    EXPECT_EQ(dc0::ENCODING_SIZE, 22 * 64);
}

TEST_F(BoardEncodingTest, StartingPosition) {
    fen::FEN f = fen::load_from_string(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Board board(f);
    dc0::encode_board(board, planes);

    // Plane 0: White pawns on A2-H2 (squares 8-15)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq >= 8 && sq <= 15) ? 1.0f : 0.0f;
        EXPECT_NEAR(planes[0 * 64 + sq], expected, 1e-6f) << "white pawn sq=" << sq;
    }

    // Plane 6: Black pawns on A7-H7 (squares 48-55)
    for (int sq = 0; sq < 64; ++sq) {
        float expected = (sq >= 48 && sq <= 55) ? 1.0f : 0.0f;
        EXPECT_NEAR(planes[6 * 64 + sq], expected, 1e-6f) << "black pawn sq=" << sq;
    }

    // Plane 5: White king on E1 (square 4)
    EXPECT_NEAR(planes[5 * 64 + 4], 1.0f, 1e-6f);
    int king_count = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (planes[5 * 64 + sq] > 0.5f) ++king_count;
    }
    EXPECT_EQ(king_count, 1);

    // Plane 11: Black king on E8 (square 60)
    EXPECT_NEAR(planes[11 * 64 + 60], 1.0f, 1e-6f);

    // Plane 12-13: repetition (should be 0 - novel position)
    for (int sq = 0; sq < 64; ++sq) {
        EXPECT_NEAR(planes[12 * 64 + sq], 0.0f, 1e-6f);
        EXPECT_NEAR(planes[13 * 64 + sq], 0.0f, 1e-6f);
    }

    // Plane 14: white to move = all 1s
    EXPECT_NEAR(planes[14 * 64 + 0], 1.0f, 1e-6f);
    EXPECT_NEAR(planes[14 * 64 + 63], 1.0f, 1e-6f);

    // Plane 15: fullmove = 1, so 1/100 = 0.01
    EXPECT_NEAR(planes[15 * 64 + 0], 0.01f, 1e-6f);

    // Planes 16-19: all castling rights available
    for (int p = 16; p <= 19; ++p) {
        EXPECT_NEAR(planes[p * 64 + 0], 1.0f, 1e-6f);
    }

    // Plane 20: no en passant (all zeros)
    for (int sq = 0; sq < 64; ++sq) {
        EXPECT_NEAR(planes[20 * 64 + sq], 0.0f, 1e-6f);
    }

    // Plane 21: halfmove clock = 0
    EXPECT_NEAR(planes[21 * 64 + 0], 0.0f, 1e-6f);
}

TEST_F(BoardEncodingTest, AfterE4) {
    fen::FEN f = fen::load_from_string(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    Board board(f);
    dc0::encode_board(board, planes);

    // White pawn on E4 (square 28) and NOT on E2 (square 12)
    EXPECT_NEAR(planes[0 * 64 + 28], 1.0f, 1e-6f);
    EXPECT_NEAR(planes[0 * 64 + 12], 0.0f, 1e-6f);

    // Plane 14: black to move = all 0s
    EXPECT_NEAR(planes[14 * 64 + 0], 0.0f, 1e-6f);

    // Plane 20: en passant on E3 (square 20)
    EXPECT_NEAR(planes[20 * 64 + 20], 1.0f, 1e-6f);
    int ep_count = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (planes[20 * 64 + sq] > 0.5f) ++ep_count;
    }
    EXPECT_EQ(ep_count, 1);
}

TEST_F(BoardEncodingTest, NoCastling) {
    fen::FEN f = fen::load_from_string(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    Board board(f);
    dc0::encode_board(board, planes);

    for (int p = 16; p <= 19; ++p) {
        EXPECT_NEAR(planes[p * 64 + 0], 0.0f, 1e-6f);
    }
}

TEST_F(BoardEncodingTest, HalfmoveAndFullmove) {
    fen::FEN f = fen::load_from_string(
        "4k3/8/8/8/8/8/8/4K3 w - - 30 50");
    Board board(f);
    dc0::encode_board(board, planes);

    // Fullmove = 50, plane value = 50/100 = 0.5
    EXPECT_NEAR(planes[15 * 64 + 0], 0.5f, 1e-6f);

    // Halfmove clock = 30, plane value = 30/100 = 0.3
    EXPECT_NEAR(planes[21 * 64 + 0], 0.3f, 1e-4f);
}

TEST_F(BoardEncodingTest, LegalMoveMask) {
    fen::FEN f = fen::load_from_string(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Board board(f);

    bool mask[dc0::POLICY_SIZE];
    dc0::encode_legal_moves(board, mask);

    int legal_count = 0;
    for (int i = 0; i < dc0::POLICY_SIZE; ++i) {
        if (mask[i]) ++legal_count;
    }
    EXPECT_EQ(legal_count, 20);
}

TEST_F(BoardEncodingTest, LegalMovesPromotion) {
    // White pawn on E7, black king on D8
    fen::FEN f = fen::load_from_string(
        "3k4/4P3/8/8/8/8/8/4K3 w - - 0 1");
    Board board(f);

    bool mask[dc0::POLICY_SIZE];
    dc0::encode_legal_moves(board, mask);

    // e7e8=N, e7e8=B, e7e8=R, e7e8=Q should all be legal
    int promo_e8 = 0;
    for (PIECE pp : {KNIGHT, BISHOP, ROOK, QUEEN}) {
        int idx = dc0::encode_move(52, 60, pp);  // E7=52, E8=60
        if (idx >= 0 && mask[idx]) ++promo_e8;
    }
    EXPECT_EQ(promo_e8, 4);

    // e7xd8=N/B/R/Q should also be legal (capture on d8)
    int promo_d8 = 0;
    for (PIECE pp : {KNIGHT, BISHOP, ROOK, QUEEN}) {
        int idx = dc0::encode_move(52, 59, pp);  // E7=52, D8=59
        if (idx >= 0 && mask[idx]) ++promo_d8;
    }
    EXPECT_EQ(promo_d8, 4);
}

// Non-pawn moves to the back rank must encode correctly.
TEST_F(BoardEncodingTest, NonPawnBackRankMove) {
    // King on e7 can capture queen on e8
    fen::FEN f = fen::load_from_string(
        "1nb1Q3/4k1rp/2ppp2B/PN6/P1P1P2P/1K1P4/4BR2/1N6 b - - 2 84");
    Board board(f);
    ASSERT_TRUE(board.can_move());

    bool mask[dc0::POLICY_SIZE];
    dc0::encode_legal_moves(board, mask);

    int count = 0;
    for (int i = 0; i < dc0::POLICY_SIZE; ++i) if (mask[i]) count++;
    EXPECT_GE(count, 1) << "Ke7xe8 must be encodable";

    // The specific move e7->e8 uses the queen-promotion encoding slot
    int idx = dc0::encode_move(52, 60, QUEEN);
    ASSERT_GE(idx, 0);
    EXPECT_TRUE(mask[idx]) << "Ke7xe8 should be in the legal mask";
}

// Perft-walk: verify encode_legal_moves count matches board move count
// for every position reachable from starting position up to depth 3.
namespace {
int perft_encoding_check(Board& board, int depth, int& mismatches) {
    if (depth == 0) return 1;
    if (board.is_checkmate() || board.is_draw()) return 1;

    // Count board-legal moves
    int board_moves = 0;
    for (int sq = 0; sq < board::SIZE; ++sq) {
        bitmask::foreach(board.state.moves[sq], [&](pos_t) noexcept -> void {
            ++board_moves;
        });
    }

    // Count encoded moves
    bool mask[dc0::POLICY_SIZE] = {};
    dc0::encode_legal_moves(board, mask);
    int encoded_moves = 0;
    for (int i = 0; i < dc0::POLICY_SIZE; ++i) if (mask[i]) encoded_moves++;

    // Promotions: board has 1 move, encoding has 4 (N/B/R/Q).
    int promo_count = 0;
    piece_bitboard_t pawns = board.get_mask(Piece(PAWN, board.activePlayer()));
    for (int sq = 0; sq < board::SIZE; ++sq) {
        if (!((pawns >> sq) & 1)) continue;
        int fr = sq / board::LEN;
        bitmask::foreach(board.state.moves[sq], [&](pos_t to) noexcept -> void {
            int tr = to / board::LEN;
            if ((tr == 7 && fr == 6) || (tr == 0 && fr == 1)) promo_count++;
        });
    }

    if (encoded_moves != board_moves + promo_count * 3) {
        mismatches++;
    }

    // Recurse
    int nodes = 0;
    for (int sq = 0; sq < board::SIZE; ++sq) {
        piece_bitboard_t m = board.state.moves[sq];
        if (!m) continue;
        bool is_pawn = (pawns >> sq) & 1;
        int fr = sq / board::LEN;
        bitmask::foreach(m, [&](pos_t to) noexcept -> void {
            int tr = to / board::LEN;
            bool is_promo = is_pawn &&
                ((tr == 7 && fr == 6) || (tr == 0 && fr == 1));

            if (is_promo) {
                for (PIECE pp : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                    move_t mv = (sq << 8) | (to & board::MOVEMASK)
                              | ((pp == QUEEN) ? board::PROMOTE_QUEEN :
                                  (pp == ROOK) ? board::PROMOTE_ROOK :
                                  (pp == BISHOP) ? board::PROMOTE_BISHOP :
                                  board::PROMOTE_KNIGHT);
                    board.make_move(mv);
                    nodes += perft_encoding_check(board, depth - 1, mismatches);
                    board.retract_move();
                }
            } else {
                move_t mv = (sq << 8) | (to & board::MOVEMASK);
                board.make_move(mv);
                nodes += perft_encoding_check(board, depth - 1, mismatches);
                board.retract_move();
            }
        });
    }
    return nodes;
}
} // namespace

TEST_F(BoardEncodingTest, PerftEncodingConsistency) {
    // Positions from scripts/perft_test.py — covers promotions, en passant,
    // castling, checks, and other tricky edge cases.
    struct Case { const char* fen; int depth; };
    Case cases[] = {
        {nullptr, 3},  // starting position
        // Kiwipete: lots of captures, castling, en passant
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0", 3},
        // Promotions and en passant heavy
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 0", 4},
        // Queen vs pawns endgame
        {"8/8/q7/8/3kp1QP/6P1/5P1K/8 w - - 0 82", 3},
        // Many promotions
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3},
        // Promotion with capture
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3},
        // Symmetric middlegame
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3},
        // Promotion on g8 with capture
        {"r2qk1nr/2pn2Pp/8/8/2P5/8/4PP1P/1Nb1KBNb w kq - 0 10", 3},
        // King + pawn endgame with promos
        {"8/kp4P1/p3K3/4P3/3p1P2/4nr2/8/8 w - - 1 55", 3},
        // En passant edge case
        {"rn1k1b1r/ppp3pp/8/4q3/2PPpR2/1P6/1P4PP/RNBQ2K1 b - d3 0 12", 3},
    };

    for (auto& c : cases) {
        Board board = c.fen ? Board(fen::load_from_string(c.fen))
                            : Board(fen::starting_pos);
        int mismatches = 0;
        int nodes = perft_encoding_check(board, c.depth, mismatches);
        EXPECT_EQ(mismatches, 0)
            << "encoding mismatch in " << mismatches << " positions for FEN: "
            << (c.fen ? c.fen : "startpos") << " (nodes=" << nodes << ")";
    }
}
