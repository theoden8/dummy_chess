#pragma once


#include <FEN.hpp>
#include <Board.hpp>


class Engine : public Board {
public:
  Engine(const fen::FEN fen=fen::starting_pos):
    Board(fen)
  {}

  template <typename F>
  inline void iter_moves_from(pos_t i, F &&func) const {
    bitmask::foreach(get_moves_from(i), [&](pos_t j) mutable -> void {
      if(is_promotion_move(i, j)) {
        for(pos_t promotion : {board::PROMOTE_KNIGHT, board::PROMOTE_BISHOP,
                               board::PROMOTE_ROOK, board::PROMOTE_QUEEN})
        {
          func(i, j | promotion);
        }
      } else {
        func(i, j);
      }
    });
  }

  template <typename F>
  inline void iter_moves(F &&func) const {
    for(pos_t i = 0; i < board::SIZE; ++i) {
      iter_moves_from(i, func);
    }
  }

  move_t get_random_move() const {
    std::vector<move_t> moves;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      moves.push_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nomove;
    return moves[rand() % moves.size()];
  }

  move_t get_random_move_from(pos_t i) const {
    std::vector<move_t> moves;
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      moves.push_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty()) {
      return board::nomove;
    }
    return moves[rand() % moves.size()];
  }
};
