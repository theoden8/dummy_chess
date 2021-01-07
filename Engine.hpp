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

  double h_material(COLOR c) const {
    const double material_pawn = 1,
                 material_knight = 3,
                 material_bishop = 3,
                 material_rook = 5,
                 material_queen = 10,
                 material_king = 1e9;
    return get_piece(PAWN,c).size() * material_pawn
         + get_piece(KNIGHT,c).size() * material_knight
         + get_piece(BISHOP,c).size() * material_bishop
         + get_piece(ROOK,c).size() * material_rook
         + get_piece(QUEEN,c).size() * material_queen;
  }

  double h_pins(COLOR c) const {
    return bitmask::count_bits(state_pins[c]);
  }

  double h_attack_cells(COLOR c) const {
    double attacks = 0;
    for(auto ac:state_attacks_count[c]) {
      attacks += ac;
    }
    return attacks;
  }

  double heuristic_of(COLOR c) const {
    double h = 0;
    h += h_material(c);
    h += h_pins(c) / 10;
    h += h_attack_cells(c) / 1000;
    return h;
  }

  double heuristic() const {
    const COLOR c = activePlayer();
    double material = heuristic_of(c) - heuristic_of(enemy_of(c));
    return material;
  }

  void _get_fixed_depth_move(pos_t depth, double &alpha, move_t &m) {
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      event_t ev = get_move_event(i, j);
      act_event(ev);
      if(depth == 1) {
        auto h = heuristic();
        if(h > alpha) {
          alpha = h;
          m = bitmask::_pos_pair(i, j);
        }
      } else {
        double cur_alpha = alpha;
        _get_fixed_depth_move(depth - 1, alpha, m);
        if(cur_alpha < alpha) {
          m = bitmask::_pos_pair(i, j);
          cur_alpha = alpha;
        }
      }
      unact_event();
    });
  }

  move_t get_fixed_depth_move(pos_t depth=1) {
    move_t m = board::nomove;
    double alpha = -1e9;
    _get_fixed_depth_move(depth, alpha, m);
    return m;
  }
};
