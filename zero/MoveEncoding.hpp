#pragma once

// AlphaZero move encoding for dummy_chess.
//
// Encodes chess moves as (from_square, move_type) -> policy index.
// Total policy size: 64 * 73 = 4672.
//
// Move types (73):
//   0-55:  Queen-like moves: 8 directions x 7 distances
//   56-63: Knight moves: 8 L-shaped offsets
//   64-72: Underpromotions: 3 directions x 3 piece types (N, B, R)
//          Queen promotions are encoded as regular queen moves.
//
// Board layout (matching dummy_chess Piece.hpp):
//   A1=0, B1=1, ..., H1=7, A2=8, ..., H8=63
//   file = sq % 8, rank = sq / 8

#include <array>
#include <cstdint>

#include <Piece.hpp>

namespace dc0 {

static constexpr int NUM_MOVE_TYPES = 73;
static constexpr int POLICY_SIZE = board::SIZE * NUM_MOVE_TYPES;  // 4672

// Decoded move: (from_sq, to_sq, promo_piece)
// promo_piece: EMPTY for non-promotions, KNIGHT/BISHOP/ROOK/QUEEN for promotions
struct DecodedMove {
    pos_t from_sq;
    pos_t to_sq;
    PIECE promo;
};

namespace detail {

// 8 queen/king directions: {dfile, drank}
static constexpr int QUEEN_DIRS[8][2] = {
    {0, 1}, {1, 1}, {1, 0}, {1, -1},
    {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
};

// 8 knight offsets: {dfile, drank}
static constexpr int KNIGHT_OFFSETS[8][2] = {
    {1, 2}, {2, 1}, {2, -1}, {1, -2},
    {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
};

// Underpromotion dfile offsets: left-capture, forward, right-capture
static constexpr int UNDERPROMO_DFILES[3] = {-1, 0, 1};

// Underpromotion pieces (not queen — queen promo is a queen-move-type)
static constexpr PIECE UNDERPROMO_PIECES[3] = {KNIGHT, BISHOP, ROOK};

static constexpr int file_of(int sq) { return sq % 8; }
static constexpr int rank_of(int sq) { return sq / 8; }
static constexpr int sq_of(int file, int rank) { return file + rank * 8; }
static constexpr bool on_board(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

// Encode table: [from_sq][to_sq][promo_idx] -> policy index (-1 if invalid)
// promo_idx: 0=none, 1=knight, 2=bishop, 3=rook, 4=queen
static constexpr int PROMO_NONE = 0;
static constexpr int PROMO_KNIGHT = 1;
static constexpr int PROMO_BISHOP = 2;
static constexpr int PROMO_ROOK = 3;
static constexpr int PROMO_QUEEN = 4;

static constexpr int piece_to_promo_idx(PIECE p) {
    switch (p) {
        case KNIGHT: return PROMO_KNIGHT;
        case BISHOP: return PROMO_BISHOP;
        case ROOK:   return PROMO_ROOK;
        case QUEEN:  return PROMO_QUEEN;
        default:     return PROMO_NONE;
    }
}

static constexpr PIECE promo_idx_to_piece(int idx) {
    switch (idx) {
        case PROMO_KNIGHT: return KNIGHT;
        case PROMO_BISHOP: return BISHOP;
        case PROMO_ROOK:   return ROOK;
        case PROMO_QUEEN:  return QUEEN;
        default:           return EMPTY;
    }
}

struct Tables {
    // encode_table[from_sq][to_sq][promo_idx] -> policy index, -1 if invalid
    int16_t encode_table[64][64][5];
    // decode_table[policy_idx] -> DecodedMove
    DecodedMove decode_table[POLICY_SIZE];
    // valid[policy_idx] -> true if this slot maps to a real move
    bool valid[POLICY_SIZE];

    constexpr Tables() : encode_table{}, decode_table{}, valid{} {
        // Initialize encode table to -1
        for (int i = 0; i < 64; ++i)
            for (int j = 0; j < 64; ++j)
                for (int k = 0; k < 5; ++k)
                    encode_table[i][j][k] = -1;

        // Initialize decode table
        for (int i = 0; i < POLICY_SIZE; ++i) {
            decode_table[i] = {board::nopos, board::nopos, EMPTY};
            valid[i] = false;
        }

        // Queen/king moves: 8 directions x 7 distances
        for (int from_sq = 0; from_sq < 64; ++from_sq) {
            int ff = file_of(from_sq);
            int fr = rank_of(from_sq);

            for (int dir = 0; dir < 8; ++dir) {
                int df = QUEEN_DIRS[dir][0];
                int dr = QUEEN_DIRS[dir][1];
                for (int dist = 1; dist <= 7; ++dist) {
                    int tf = ff + df * dist;
                    int tr = fr + dr * dist;
                    if (!on_board(tf, tr)) break;
                    int to_sq = sq_of(tf, tr);
                    int mt = dir * 7 + (dist - 1);
                    int policy_idx = from_sq * NUM_MOVE_TYPES + mt;

                    // Check promotion: pawn reaching last rank with dist=1
                    bool white_promo = (fr == 6 && tr == 7 && dist == 1);
                    bool black_promo = (fr == 1 && tr == 0 && dist == 1);
                    if (white_promo || black_promo) {
                        // This slot encodes queen promotion
                        encode_table[from_sq][to_sq][PROMO_QUEEN] = (int16_t)policy_idx;
                        decode_table[policy_idx] = {(pos_t)from_sq, (pos_t)to_sq, QUEEN};
                    } else {
                        encode_table[from_sq][to_sq][PROMO_NONE] = (int16_t)policy_idx;
                        decode_table[policy_idx] = {(pos_t)from_sq, (pos_t)to_sq, EMPTY};
                    }
                    valid[policy_idx] = true;
                }
            }

            // Knight moves
            for (int off = 0; off < 8; ++off) {
                int tf = ff + KNIGHT_OFFSETS[off][0];
                int tr = fr + KNIGHT_OFFSETS[off][1];
                if (!on_board(tf, tr)) continue;
                int to_sq = sq_of(tf, tr);
                int mt = 56 + off;
                int policy_idx = from_sq * NUM_MOVE_TYPES + mt;
                encode_table[from_sq][to_sq][PROMO_NONE] = (int16_t)policy_idx;
                decode_table[policy_idx] = {(pos_t)from_sq, (pos_t)to_sq, EMPTY};
                valid[policy_idx] = true;
            }

            // Underpromotions (only from rank 6->7 or rank 1->0)
            for (int promo_from : {6, 1}) {
                if (fr != promo_from) continue;
                int promo_to = (promo_from == 6) ? 7 : 0;
                for (int dir_idx = 0; dir_idx < 3; ++dir_idx) {
                    int tf = ff + UNDERPROMO_DFILES[dir_idx];
                    if (!on_board(tf, promo_to)) continue;
                    int to_sq = sq_of(tf, promo_to);
                    for (int piece_idx = 0; piece_idx < 3; ++piece_idx) {
                        int mt = 64 + dir_idx * 3 + piece_idx;
                        int policy_idx = from_sq * NUM_MOVE_TYPES + mt;
                        int promo_idx = piece_to_promo_idx(UNDERPROMO_PIECES[piece_idx]);
                        encode_table[from_sq][to_sq][promo_idx] = (int16_t)policy_idx;
                        decode_table[policy_idx] = {
                            (pos_t)from_sq, (pos_t)to_sq, UNDERPROMO_PIECES[piece_idx]
                        };
                        valid[policy_idx] = true;
                    }
                }
            }
        }
    }
};

// Compile-time table generation
static constexpr Tables TABLES = Tables();

} // namespace detail


// Encode a move to policy index.
// from_sq, to_sq: square indices 0-63
// promo: EMPTY for non-promotions, KNIGHT/BISHOP/ROOK/QUEEN for promotions
inline int encode_move(pos_t from_sq, pos_t to_sq, PIECE promo = EMPTY) {
    int promo_idx = detail::piece_to_promo_idx(promo);
    return detail::TABLES.encode_table[from_sq][to_sq][promo_idx];
}

// Decode a policy index to a move.
inline DecodedMove decode_move(int policy_idx) {
    return detail::TABLES.decode_table[policy_idx];
}

// Check if a policy index maps to a valid move.
inline bool is_valid_policy_index(int policy_idx) {
    return policy_idx >= 0 && policy_idx < POLICY_SIZE &&
           detail::TABLES.valid[policy_idx];
}

// Encode from engine move_t format.
// move_t: (from_sq << 8) | to_byte, where to_byte bits 5-0 = to_sq, bits 7-6 = promo
inline int encode_move_t(move_t m) {
    pos_t from_sq = bitmask::first(m) & board::MOVEMASK;
    pos_t to_sq = bitmask::second(m) & board::MOVEMASK;
    pos_t to_byte = bitmask::second(m);

    // Determine if promotion by checking if pawn reaches last rank
    int from_rank = detail::rank_of(from_sq);
    int to_rank = detail::rank_of(to_sq);
    bool is_promo = (to_rank == 7 && from_rank == 6) ||
                    (to_rank == 0 && from_rank == 1);

    if (is_promo) {
        PIECE promo = board::get_promotion_as(to_byte);
        return encode_move(from_sq, to_sq, promo);
    }
    return encode_move(from_sq, to_sq, EMPTY);
}

// Decode policy index to engine move_t format.
inline move_t decode_to_move_t(int policy_idx) {
    DecodedMove dm = decode_move(policy_idx);
    pos_t to_byte = dm.to_sq;
    if (dm.promo != EMPTY) {
        switch (dm.promo) {
            case KNIGHT: to_byte |= board::PROMOTE_KNIGHT; break;
            case BISHOP: to_byte |= board::PROMOTE_BISHOP; break;
            case ROOK:   to_byte |= board::PROMOTE_ROOK; break;
            case QUEEN:  to_byte |= board::PROMOTE_QUEEN; break;
            default: break;
        }
    }
    return bitmask::_pos_pair(dm.from_sq, to_byte);
}

} // namespace dc0
