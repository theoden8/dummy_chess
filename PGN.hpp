#pragma once


#include <string>
#include <vector>

#include <Event.hpp>
#include <Board.hpp>


namespace pgn {

constexpr bool LICHESS_COMPATIBILITY = 1;

struct PGN {
  Board &board;
  size_t cur_ply = 0;
  std::vector<std::string> ply;
  std::string ending = "";

  PGN(Board &board):
    board(board)
  {}

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
    board[i].foreach([&](pos_t k) mutable -> void {
      if(rank_resolved && file_resolved)return;
      if(!(imask & (1ULL << k)))return;
      if(board.get_moves_from(k) & jmask) {
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

  void write_event(event_t ev) {
    std::string p;
    const uint8_t marker = event::extract_marker(ev);
    switch(marker) {
      case event::BASIC_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          const pos_t killwhat = event::extract_piece_ind(ev);
          p = "";
          // pawn takes pawn (not en-passant)
          if(board[i].value==PAWN && board[j].value==PAWN && killwhat!=event::killnothing && !LICHESS_COMPATIBILITY) {
            p += resolve_ambiguity(i, j);
            p += name_of_file(j);
          } else {
            p += piece_name(board[i]);
            p += resolve_ambiguity(i, j);
            if(killwhat!=event::killnothing)p+='x';
            p += board::_pos_str(j);
          }
        }
      break;
      case event::CASTLING_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          if(board::_x(j) == C) {
            p = "O-O-O"s;
          } else {
            p = "O-O"s;
          }
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          p += resolve_ambiguity(i, j, true);
          if(LICHESS_COMPATIBILITY) {
            p += 'x';
            p += board::_pos_str(j);
          } else {
            p += name_of_file(j);
          }
        }
      break;
      case event::PROMOTION_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      to_byte = bitmask::second(m);
            const pos_t j = to_byte & board::MOVEMASK;
            const PIECE becomewhat = board::get_promotion_as(to_byte);
          const pos_t killwhat = event::extract_piece_ind(ev);
          if(killwhat != event::killnothing) {
            p += resolve_ambiguity(i, j);
            p += 'x';
          }
          p += board::_pos_str(j);
          p += '=';
          p += toupper(board.pieces[Piece::get_piece_index(becomewhat, board.activePlayer())].str());
        }
      break;
    }
    ++cur_ply;
    ply.emplace_back(p);
  }

  void handle_event(event_t ev) {
    write_event(ev);
    board.act_event(ev);
    const COLOR c = board.activePlayer();
    const pos_t no_checks = board.get_attack_counts_to(board.get_king_pos(c), enemy_of(c));
    ending = "";
    if(board.is_draw_stalemate()) {
      ending = "1/2 - 1/2 (stalemate)";
    } else if(board.is_draw_halfmoves()) {
      ending = "1/2 - 1/2 (50 moves)";
    } else if(board.is_draw_material()) {
      ending = "1/2 - 1/2 (material)";
    } else if(board.is_checkmate() && !board.can_move()) {
      ply.back() += '#';
      ending = (c == WHITE) ? "1-0"s : "0-1"s;
    } else if(no_checks > 0) {
      ply.back()+='+';
      if(no_checks>1)ply.back()+='+';
    }
  }

  void handle_move(move_t m) {
    handle_event(board.get_move_event(m));
  }

  void retract_event() {
    if(cur_ply != 0) {
      --cur_ply;
      ply.pop_back();
      ending = ""s;
    }
  }

  std::string str() {
    std::string s;
    for(size_t i = 0; i < cur_ply; ++i) {
      if(!(i & 1)) {
        s += " "s + std::to_string(i / 2 + 1) + "."s;
      }
      s += " "s + ply[i];
    }
    return s;
  }
};

} // namespace pgn
