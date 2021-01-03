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

  PGN(Board &board):
    board(board)
  {}

  static std::string write_pos(int i) {
    std::string p;
    p += 'a' + board::_x(i);
    p += '1' + board::_y(i);
    return p;
  }

  size_t size() const {
    return cur_ply;
  }

  void write_event(event_t ev) {
    std::string p;
    pos_t marker = event::extract_byte(ev);
    switch(marker) {
      case event::BASIC_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t killwhat = event::extract_byte(ev);
          p = "";
          if(board[i].value==PAWN && board[j].value==PAWN && killwhat!=event::killnothing) {
            p += 'a' + board::_x(i);
            p += 'a' + board::_x(j);
          } else {
            if(board[i].value!=PAWN)p+=toupper(board[i].str());
            if(killwhat!=event::killnothing)p+='x';
            p += write_pos(j);
          }
        }
      break;
      case event::CASTLING_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          if(board::_x(j) == C) {
            p = "O-O-O";
          } else {
            p = "O-O";
          }
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          p += 'a' + board::_x(i);
          p += 'x';
          p += write_pos(j);
        }
      break;
      case event::PROMOTION_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t killwhat = event::extract_byte(ev);
          auto castlings_ = event::extract_castlings(ev);
          auto enpassant_ = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);
          pos_t becomewhat = event::extract_byte(ev);
          if(killwhat != event::killnothing) {
            p += 'a' + board::_x(i);
            p += 'x';
          }
          p += write_pos(j);
          p += '=';
          p += toupper(board.pieces[becomewhat].str());
        }
      break;
    }
    ++cur_ply;
    ply.push_back(p);
  }

  void retract_event() {
    if(cur_ply != 0) {
      --cur_ply;
      ply.pop_back();
    }
  }
};

} // namespace pgn
