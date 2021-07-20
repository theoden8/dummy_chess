#pragma once


#include <string>
#include <vector>

#include <Board.hpp>
#include <MoveLine.hpp>


namespace pgn {

constexpr bool LICHESS_COMPATIBILITY = 1;

struct PGN {
  fen::FEN startfen;
  Board &board;
  size_t cur_ply = 0;
  std::vector<std::string> ply;
  std::string ending = "";

  PGN(Board &board):
    board(board)
  {
    startfen = board.export_as_fen();
  }

  size_t size() const {
    return cur_ply;
  }

  char name_of_file(pos_t i) const {
    return 'a' + board::_x(i);
  }

  char name_of_rank(pos_t i) const {
    return '1' + board::_y(i);
  }

  std::string piece_name(Piece p) {
    std::string s = ""s;
    if(p.value == PAWN)return s;
    s += toupper(p.str());
    return s;
  }

  std::string resolve_ambiguity(pos_t i, pos_t j, bool enpassant=false) const {
    std::string resolve = ""s;
    piece_bitboard_t imask = ~0uLL,
                     jmask = 1ULL << j;
    if(board[i].value == PAWN && (board[i].value == PAWN || board[i].value == EMPTY)) {
      if(board[j].value != EMPTY || enpassant) {
        resolve += name_of_file(i);
      }
      imask = board::file_mask(board::_x(i));
      if(!LICHESS_COMPATIBILITY) {
        jmask = board::file_mask(board::_x(j));
      }
    }
    bool file_resolved = false, rank_resolved = false;

    piece_bitboard_t mask = ~0x00ULL;
    const COLOR c = board[i].color;
    if(board.bits_pawns & piece::pos_mask(i)) {
      mask = board.bits_pawns;
    } else if(board.bits_slid_diag & board.bits_slid_orth & piece::pos_mask(i)) {
      mask = board.bits_slid_diag & board.bits_slid_orth;
    } else if(board.bits_slid_diag & piece::pos_mask(i)) {
      mask = board.bits_slid_diag & ~board.bits_slid_orth;
    } else if(board.bits_slid_orth & piece::pos_mask(i)) {
      mask = board.bits_slid_orth & ~board.bits_slid_diag;
    } else if(i == board.pos_king[c]) {
      mask = piece::pos_mask(i);
    } else {
      mask = board.get_knight_bits();
    }
    mask &= board.bits[c];

    bitmask::foreach(mask, [&](pos_t k) mutable -> void {
      if(rank_resolved && file_resolved)return;
      if(!(imask & (1ULL << k)))return;
      if(board.state.moves[k] & jmask) {
        if(!file_resolved && board::_x(k) != board::_x(i)) {
          resolve += name_of_file(i);
          file_resolved = true;
        } else if(!rank_resolved && board::_y(k) != board::_y(i)) {
          resolve += name_of_rank(i);
          rank_resolved = true;
        }
      }
    });
    return resolve;
  }

  void write_move(pos_t i, pos_t j) {
    const move_t m = bitmask::_pos_pair(i, j);
    const pos_t promote_as = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    std::string p;
    if(m == board::nullmove) {
      p = "0000"s;
    } else if(board.is_drop_move(i, j)) {
      p = board::_move_str(bitmask::_pos_pair(i, j));
    } else if(board.is_castling_move(i, j)) {
      const COLOR c = board.color_at_pos(i);
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      pos_t k_j = j;
      if(board.chess960) {
        if(j == board::_pos(board.qcastlrook[c], castlrank)) {
          k_j = board::_pos(C, castlrank);
        } else if(j == board::_pos(board.kcastlrook[c], castlrank)) {
          k_j = board::_pos(G, castlrank);
        } else {
          abort();
        }
      }
      const move_t rookmove = piece::get_king_castle_rook_move(c, i, k_j, board.qcastlrook[c], board.kcastlrook[c]);
      const pos_t r_i = bitmask::first(rookmove),
                  r_j = bitmask::second(rookmove);
      if(board::_x(r_i) == board.qcastlrook[c]) {
        p = "O-O-O"s;
      } else {
        p = "O-O"s;
      }
    } else if(board.is_enpassant_take_move(i, j)) {
      const pos_t killwhere = board.enpassant_pawn();
      p += resolve_ambiguity(i, j, true);
      if(LICHESS_COMPATIBILITY) {
        p += 'x';
        p += board::_pos_str(j);
      } else {
        p += name_of_file(j);
      }
    } else if(board.is_promotion_move(i, j)) {
      const PIECE becomewhat = board::get_promotion_as(promote_as);
      const bool is_capture = !board.empty_at_pos(j);
      if(is_capture) {
        p += resolve_ambiguity(i, j);
        p += 'x';
      }
      p += board::_pos_str(j);
      p += '=';
      p += toupper(Piece(becomewhat, board.activePlayer()).str());
    } else {
      const bool is_capture = !board.empty_at_pos(j);
      p = "";
      // pawn takes pawn (not en-passant)
      if(board[i].value==PAWN && board[j].value==PAWN && is_capture && !LICHESS_COMPATIBILITY) {
        p += resolve_ambiguity(i, j);
        p += name_of_file(j);
      } else {
        p += piece_name(board[i]);
        p += resolve_ambiguity(i, j);
        if(is_capture)p+='x';
        p += board::_pos_str(j);
      }
    }
    ++cur_ply;
    ply.emplace_back(p);
  }

  INLINE void write_move(move_t m) {
    write_move(bitmask::first(m), bitmask::second(m));
  }

  void handle_move(move_t m) {
    assert(m == board::nullmove || board.check_valid_move(m));
    write_move(m);
    board.make_move(m);
    const COLOR c = board.activePlayer();
    const pos_t no_checks = board.get_attack_counts_to(board.pos_king[c], enemy_of(c));
    ending = "";
    if(board.is_draw_stalemate()) {
      ending = "1/2 - 1/2 (stalemate)";
    } else if(board.is_draw_halfmoves()) {
      ending = "1/2 - 1/2 (50 moves)";
    } else if(board.is_draw_material()) {
      ending = "1/2 - 1/2 (material)";
    } else if(board.is_checkmate()) {
      ply.back() += '#';
      ending = (c == WHITE) ? "1-0"s : "0-1"s;
    } else if(no_checks > 0) {
      ply.back()+='+';
      if(no_checks>1)ply.back()+='+';
    } else if(board.can_draw_repetition()) {
      ending = "1/2 - 1/2 (repetitions)";
    }
  }

  void handle_move(pos_t i, pos_t j) {
    handle_move(bitmask::_pos_pair(i, j));
  }

  void retract_move() {
    if(cur_ply != 0) {
      board.retract_move();
      --cur_ply;
      ply.pop_back();
      ending = ""s;
    }
  }

  NEVER_INLINE std::string str() const {
    std::string s;
    if(startfen != fen::starting_pos) {
      s += "[FEN] \""s + fen::export_as_string(startfen) + "\"]\n\n";
      s += "[FEN] \""s + fen::export_as_string(fen::starting_pos) + "\"]\n\n";
    }
    for(size_t i = 0; i < cur_ply; ++i) {
      if(!(i & 1)) {
        s += " "s + std::to_string(i / 2 + 1) + "."s;
      }
      s += " "s + ply[i];
    }
    return s;
  }
};

std::string _move_str(Board &b, move_t m) {
  assert(b.check_valid_move(m));
  pgn::PGN pgn(b);
  pgn.handle_move(m);
  std::string s = pgn.ply.front();
  pgn.retract_move();
  return s;
}

NEVER_INLINE std::string _line_str(Board &b, const MoveLine &mline) {
  assert(b.check_valid_sequence(mline));
  pgn::PGN pgn(b);
  for(auto m : mline) {
    pgn.handle_move(m);
  }
  std::string s = str::join(pgn.ply, " "s);
  for(auto m : mline) {
    pgn.retract_move();
  }
  return s;
}

NEVER_INLINE std::string _line_str_full(Board &b, const MoveLine &mline) {
  assert(b.check_valid_sequence(mline));
  pgn::PGN pgn(b);
  for(auto m : mline.get_past()) {
    b.retract_move();
  }
  for(auto m : mline.get_past()) {
    pgn.handle_move(m);
  }
  std::string s = ""s;
  if(mline.start > 0) {
    s += "["s + str::join(pgn.ply, " "s) + "]"s;
  }
  if(!mline.empty()) {
    s += " "s + _line_str(b, mline.get_future());
  }
  return s;
}

} // namespace pgn
