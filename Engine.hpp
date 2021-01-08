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

  const double material_pawn = 1,
               material_knight = 3,
               material_bishop = 3,
               material_rook = 5,
               material_queen = 10,
               material_king = 1e9;

  double material_of(PIECE p) const {
    switch(p) {
      case EMPTY:return 0;
      case PAWN:return material_pawn;
      case KNIGHT:return material_knight;
      case BISHOP:return material_bishop;
      case ROOK:return material_rook;
      case QUEEN:return material_queen;
      case KING:return material_king;
    }
    return 0;
  }

  double h_material(COLOR c) const {
    double m = 0;
    for(PIECE p : {PAWN,KNIGHT,BISHOP,ROOK,QUEEN}) {
      m += get_piece(p,c).size() * material_of(p);
    }
    return m;
  }

  double h_pins(COLOR c) const {
    double h = 0;
    bitmask::foreach(state_pins[enemy_of(c)], [&](pos_t i) mutable -> void {
      h += material_of(self[i].value);
    });
    return h;
  }

  double h_attack_cells(COLOR c) const {
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    double attacks = 0;
    for(pos_t i = 0; i < board::SIZE; ++i) {
      if(self[i].color != c)continue;
      auto a = self.get_attacks_from(i);
      attacks += bitmask::count_bits(a & occupied) + bitmask::count_bits(a);
    }
    return attacks;
  }

  double heuristic_of(COLOR c) const {
    double h = 0;
    if(self.is_draw())return 0;
    bool canmove = false;
    const pos_t no_checks = get_attack_counts_to(get_king_pos(c), enemy_of(c));
    for(const auto &m : state_moves)if(m){canmove=true;break;}
    if(!canmove && no_checks > 0) {
      return -1e9;
    }
    h += h_material(c);
    h += h_pins(c) / 100;
    h += h_attack_cells(c) / 10000;
    return h;
  }

  double heuristic(COLOR c) const {
    return heuristic_of(c) - heuristic_of(enemy_of(c));
  }

  void _get_fixed_depth_move(COLOR c, pos_t depth, double &alpha, move_t &m, size_t &nodes) {
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      event_t ev = get_move_event(i, j);
      act_event(ev);
      if(depth == 1) {
        auto h = heuristic(c);
        if(h > alpha) {
          alpha = h;
          m = bitmask::_pos_pair(i, j);
        }
        ++nodes;
      } else {
        double cur_alpha = alpha;
        _get_fixed_depth_move(c, depth - 1, alpha, m, nodes);
        if(cur_alpha < alpha) {
          m = bitmask::_pos_pair(i, j);
          cur_alpha = alpha;
        }
      }
      unact_event();
    });
  }

  size_t nodes_searched = 0;
  move_t get_fixed_depth_move(pos_t depth=1) {
    move_t m = board::nomove;
    double alpha = -1e9;
    nodes_searched = 0;
    _get_fixed_depth_move(activePlayer(), depth, alpha, m, nodes_searched);
    return m;
  }
};
