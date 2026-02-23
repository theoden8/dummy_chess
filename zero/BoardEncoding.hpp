#pragma once

// dc0 board encoding: Board -> (22, 8, 8) float tensor for neural network input.
//
// Plane layout (22 planes):
//   0-11:  Piece planes (6 types x 2 colors), 1.0 where piece present
//          Order: W.Pawn, W.Knight, W.Bishop, W.Rook, W.Queen, W.King,
//                 B.Pawn, B.Knight, B.Bishop, B.Rook, B.Queen, B.King
//   12:    Repetition >= 2 (all 1s if position seen at least once before)
//   13:    Repetition >= 3 (all 1s if position seen at least twice before)
//   14:    Color to move (all 1s if white, all 0s if black)
//   15:    Fullmove count / 100 (scalar broadcast to all squares)
//   16:    Castling: white kingside  (all 1s if available)
//   17:    Castling: white queenside (all 1s if available)
//   18:    Castling: black kingside  (all 1s if available)
//   19:    Castling: black queenside (all 1s if available)
//   20:    En passant (1.0 on target square, 0 elsewhere)
//   21:    Halfmove clock / 100 (scalar broadcast to all squares)
//
// Board square layout matches dummy_chess:
//   A1=0, B1=1, ..., H1=7, A2=8, ..., H8=63
//   In the tensor: dim1 = rank (0-7), dim2 = file (0-7)
//   So tensor[plane][rank][file] maps to square = file + rank * 8

#include <cstring>
#include <algorithm>
#include <array>

#include <Bitmask.hpp>
#include <Piece.hpp>
#include <Board.hpp>

#include <MoveEncoding.hpp>

namespace dc0 {

static constexpr int ENCODING_PLANES = 22;
static constexpr int ENCODING_SIZE = ENCODING_PLANES * board::SIZE;  // 22 * 64 = 1408

// Count how many times the current position has occurred in the game history.
// Returns 1 if novel, 2 if seen once before, etc.
inline int count_repetitions(const Board& board) {
    int reps = 1;
    const size_t hist_size = board.state_hist.size();
    const size_t no_iter = std::min<size_t>(hist_size, board.get_halfmoves());
    // Walk backwards in steps of 2 (same side to move)
    for (size_t i = NO_COLORS - 1; i < no_iter; i += NO_COLORS) {
        const auto& prev = board.state_hist[hist_size - i - 1];
        if (board.state.info == prev.info && !prev.null_move_state) {
            ++reps;
        }
    }
    return reps;
}

// Write a bitboard into a plane buffer.
// plane points to 64 floats, indexed by square (file + rank * 8).
inline void bitboard_to_plane(piece_bitboard_t bb, float* plane) {
    bitmask::foreach(bb, [&](pos_t sq) noexcept -> void {
        plane[sq] = 1.0f;
    });
}

// Fill a plane with a constant value.
inline void fill_plane(float* plane, float value) {
    for (int i = 0; i < 64; ++i) {
        plane[i] = value;
    }
}

// Encode a Board position into 22 * 64 = 1408 floats.
// Output layout: planes[plane_idx * 64 + square], where square = file + rank * 8.
// This matches the tensor layout (ENCODING_PLANES, 8, 8) with dim1=rank, dim2=file.
inline void encode_board(const Board& board, float* out) {
    std::memset(out, 0, ENCODING_SIZE * sizeof(float));

    // Planes 0-11: piece planes
    int plane = 0;
    for (COLOR c : {WHITE, BLACK}) {
        for (PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
            piece_bitboard_t bb = board.get_mask(Piece(p, c));
            bitboard_to_plane(bb, out + plane * 64);
            ++plane;
        }
    }

    // Planes 12-13: repetition
    int reps = count_repetitions(board);
    if (reps >= 2) fill_plane(out + 12 * 64, 1.0f);
    if (reps >= 3) fill_plane(out + 13 * 64, 1.0f);

    // Plane 14: color to move (1.0 = white, 0.0 = black)
    if (board.activePlayer() == WHITE) {
        fill_plane(out + 14 * 64, 1.0f);
    }

    // Plane 15: fullmove count / 100
    int fullmove = ((board.get_current_ply() - 1) / 2) + 1;
    fill_plane(out + 15 * 64, float(fullmove) / 100.0f);

    // Planes 16-19: castling rights
    if (board.is_castling(WHITE, KING_SIDE))  fill_plane(out + 16 * 64, 1.0f);
    if (board.is_castling(WHITE, QUEEN_SIDE)) fill_plane(out + 17 * 64, 1.0f);
    if (board.is_castling(BLACK, KING_SIDE))  fill_plane(out + 18 * 64, 1.0f);
    if (board.is_castling(BLACK, QUEEN_SIDE)) fill_plane(out + 19 * 64, 1.0f);

    // Plane 20: en passant target square
    pos_t ep = board.enpassant_trace();
    if (ep != board::nopos) {
        out[20 * 64 + ep] = 1.0f;
    }

    // Plane 21: halfmove clock / 100
    pos_t halfmoves = board.get_halfmoves();
    fill_plane(out + 21 * 64, float(halfmoves) / 100.0f);
}

// Encode legal moves as a boolean mask over the policy vector (4672 elements).
// Sets mask[policy_idx] = true for each legal move.
inline void encode_legal_moves(Board& board, bool* mask) {
    std::memset(mask, 0, POLICY_SIZE * sizeof(bool));
    // iter_moves is from Perft (Board's parent doesn't have it, but Engine/Perft does).
    // We use the Board's move/attack masks directly.
    for (pos_t from_sq = 0; from_sq < board::SIZE; ++from_sq) {
        piece_bitboard_t moves = board.state.moves[from_sq];
        if (moves == 0) continue;
        bitmask::foreach(moves, [&](pos_t to_sq) noexcept -> void {
            // Check if this is a promotion
            int from_rank = from_sq / 8;
            int to_rank = to_sq / 8;
            bool is_pawn = (board.get_mask(Piece(PAWN, board.activePlayer())) >> from_sq) & 1;
            bool is_promo = is_pawn &&
                ((to_rank == 7 && from_rank == 6) || (to_rank == 0 && from_rank == 1));

            if (is_promo) {
                // All 4 promotions are legal
                for (PIECE pp : {KNIGHT, BISHOP, ROOK, QUEEN}) {
                    int idx = encode_move(from_sq, to_sq, pp);
                    if (idx >= 0) mask[idx] = true;
                }
            } else {
                int idx = encode_move(from_sq, to_sq, EMPTY);
                if (idx >= 0) mask[idx] = true;
            }
        });
    }
}

} // namespace dc0
