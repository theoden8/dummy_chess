#pragma once


#include <string>
#include <vector>

#include <Event.hpp>
#include <Board.hpp>


namespace pgn {

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

  void write_event(event_t ev) {
    std::string p;
    pos_t marker = event::extract_byte(ev);
    switch(marker) {
      case event::BASIC_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          const pos_t killwhat = event::extract_byte(ev);
          p = "";
          if(board[i].value==PAWN && board[j].value==PAWN && killwhat!=event::killnothing) {
            p += 'a' + board::_x(i);
            p += 'a' + board::_x(j);
          } else {
            if(board[i].value!=PAWN)p+=toupper(board[i].str());
            if(killwhat!=event::killnothing)p+='x';
            p += board::_pos_str(j);
          }
        }
      break;
      case event::CASTLING_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          if(board::_x(j) == C) {
            p = "O-O-O";
          } else {
            p = "O-O";
          }
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          p += 'a' + board::_x(i);
          p += 'x';
          p += board::_pos_str(j);
        }
      break;
      case event::PROMOTION_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t to_byte = event::extract_byte(ev);
            const pos_t j = to_byte & board::MOVEMASK;
            const PIECE becomewhat = board.get_promotion_as(to_byte);
          const pos_t killwhat = event::extract_byte(ev);
          if(killwhat != event::killnothing) {
            p += 'a' + board::_x(i);
            p += 'x';
          }
          p += board::_pos_str(j);
          p += '=';
          p += toupper(board.pieces[Piece::get_piece_index(becomewhat, board.activePlayer())].str());
        }
      break;
    }
    ++cur_ply;
    ply.push_back(p);
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
    } else if(!board.can_move() && no_checks > 0) {
      ply.back() += '#';
      ending = (c == WHITE) ? "1-0"s : "0-1"s;
    } else if(no_checks > 0) {
      ply.back()+='+';
      if(no_checks>1)ply.back()+='+';
    }
  }

  void retract_event() {
    if(cur_ply != 0) {
      --cur_ply;
      ply.pop_back();
      ending = ""s;
    }
  }
};

} // namespace pgn
