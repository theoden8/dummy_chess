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
  ALWAYS_UNROLL inline void iter_moves(F &&func) const {
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

  // for testing
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
               material_queen = 9,
               material_king = 1e9;

  inline double material_of(PIECE p) const {
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

  inline double h_material(COLOR c) const {
    double m = 0;
    for(PIECE p : {PAWN,KNIGHT,BISHOP,ROOK,QUEEN}) {
      m += get_piece(p,c).size() * material_of(p);
    }
    return m;
  }

  inline double h_pins(COLOR c) const {
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
    // checkmate
    if(!can_move())return -1e9;

    h += h_material(c);
    //h += h_pins(c) / 100;
    h += h_attack_cells(c) / 1e4;
    return h;
  }

  double heuristic(COLOR c) const {
    return heuristic_of(c) - heuristic_of(enemy_of(c));
  }

  std::tuple<double, move_t> _get_fixed_depth_move(COLOR c, pos_t depth, double alpha, size_t &nodes) {
    move_t m = board::nomove;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      const event_t ev = get_move_event(i, j);
      act_event(ev);
      if(depth == 1) {
        auto h = heuristic(c);
        // me: return best one (maximize), enemy: return worst one (minimize)
        if(alpha == UNINITIALIZED || (c == activePlayer() && h < alpha) || (c != activePlayer() && h > alpha)) {
          alpha = (c == activePlayer()) ? -h : h;
          printf("new heuristic after=%s depth=%hhu alpha=%.5f\n", board::_move_str(bitmask::_pos_pair(i,j)).c_str(), depth, alpha);
          m = bitmask::_pos_pair(i, j);
        }
        ++nodes;
      } else if(depth <= 0) {
        ++nodes;
      } else {
        // me: return best one (maximize), enemy: return worst one (minimize)
        auto [new_alpha, _] = _get_fixed_depth_move(c, depth - 1, alpha, nodes);
        if(alpha == UNINITIALIZED || (c != activePlayer() && alpha < new_alpha) || (c == activePlayer() && alpha > new_alpha)) {
          alpha = new_alpha;
          m = bitmask::_pos_pair(i, j);
          printf("update move=%s depth=%hhu %s=%.5f\n", board::_move_str(m).c_str(), depth, "alpha", new_alpha);
        }
      }
      unact_event();
    });
    return std::make_tuple(alpha, m);
  }

  size_t nodes_searched = 0;
  double evaluation = 1e-9;
  const double UNINITIALIZED = -1e9;
  move_t get_fixed_depth_move(pos_t depth=1) {
    nodes_searched = 0;
    auto [alpha, m] = _get_fixed_depth_move(activePlayer(), depth, UNINITIALIZED, nodes_searched);
    printf("alpha=%.5f\n", alpha);
    evaluation = alpha;
    return m;
  }

  void _perft(pos_t depth, size_t &nodes) {
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      const event_t ev = get_move_event(i, j);
      act_event(ev);
      if(depth == 1) {
        ++nodes;
      } else if(depth <= 0) {
        ++nodes;
      } else {
        _perft(depth - 1, nodes);
      }
      unact_event();
    });
  }

  inline void perft(pos_t depth=1) {
    nodes_searched = 0;
    _perft(depth, nodes_searched);
  }
};
