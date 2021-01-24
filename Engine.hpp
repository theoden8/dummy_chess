#pragma once


#include <cfloat>
#include <algorithm>

#include <FEN.hpp>
#include <Board.hpp>


class Engine : public Board {
public:
  Engine(const fen::FEN fen=fen::starting_pos):
    Board(fen)
  {}

  template <typename F>
  INLINE void iter_moves_from(pos_t i, F &&func) const {
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
  ALWAYS_UNROLL INLINE void iter_moves(F &&func) const {
    bitmask::foreach(state_piece_positions[activePlayer()], [&](pos_t i) mutable -> void {
      iter_moves_from(i, func);
    });
  }

  INLINE size_t count_moves() const {
    size_t no_moves = 0;
    bitmask::foreach(state_piece_positions[activePlayer()], [&](pos_t i) mutable -> void {
      pos_t moves_from = bitmask::count_bits(get_moves_from(i));
      if(self[i].value == PAWN && (board::_y(i) == 2-1 || board::_y(i) == 7-1)
          && (
            (self[i].color == WHITE && board::_y(i) == 7-1)
            || (self[i].color == BLACK && board::_y(i) == 2-1)
        ))
      {
        moves_from *= 4;
      }
      no_moves += moves_from;
    });
    return no_moves;
  }

  move_t get_random_move() const {
    std::vector<move_t> moves;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nomove;
    return moves[rand() % moves.size()];
  }

  // for MC-style testing
  move_t get_random_move_from(pos_t i) const {
    std::vector<move_t> moves;
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nomove;
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
    if(!can_move())return -1e7;

    h += h_material(c);
    h += h_pins(c) / 100;
    h += h_attack_cells(c) / 1e4;
    return h;
  }

  double evaluate() const {
    const COLOR c = play_as;
    return heuristic_of(c) - heuristic_of(enemy_of(c));
  }

  struct ab_info {
    board_info info;
    pos_t depth;
    double eval;
    bool ndtype;
    move_t m;
  };
  std::pair<double, move_t> alpha_beta(double alpha, double beta, pos_t depth, std::array<ab_info, ZOBRIST_SIZE> &ab_store, bool ndtype) {
    if(depth == 0) {
      ++nodes_searched;
      return {evaluate(), board::nomove};
    }
    zobrist::key_t k = zb_hash();
    auto info = get_board_info();
    if(ab_store[k].info == info && depth <= ab_store[k].depth && ab_store[k].ndtype == ndtype) {
      ++zb_hit;
      return {ab_store[k].eval, ab_store[k].m};
    }
    bool overwrite = (depth + 1 >= ab_store[k].depth);
    ++zb_miss;
    std::vector<move_t> moves;
    moves.reserve(16);
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i,j));
    });
    move_t m_best = board::nomove;
    for(move_t m : moves) {
      make_move(m);
      double score = -alpha_beta(-beta, -alpha, depth - 1, ab_store, !ndtype).first;
      //printf("depth=%d, score=%.5f, %s\n", depth,score,board::_move_str(m).c_str());
      if(score >= beta) {
        retract_move();
        if(overwrite) {
          ab_store[k] = { .info=info, .depth=depth, .eval=beta, .ndtype=ndtype, .m=m };
        }
        return {beta, m};
      } else if(score > alpha) {
        alpha = score;
        m_best = m;
      }
      retract_move();
    };
    if(overwrite) {
      ab_store[k] = { .info=info, .depth=depth, .eval=alpha, .ndtype=ndtype, .m=m_best };
    }
    return {alpha, m_best};
  }

  std::pair<double, move_t> iterative_deepening_dfs(pos_t depth, std::array<ab_info, ZOBRIST_SIZE> &ab_store) {
    if(depth == 0)return {DBL_MAX, board::nomove};
    std::vector<move_t> moves;
    moves.reserve(16);
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i,j));
    });
    std::vector<std::pair<double, move_t>> bestmoves;
    for(auto m:moves)bestmoves.emplace_back(0.,m);
    for(pos_t d = 1; d < depth; ++d) {
      double alpha = -DBL_MAX;
      for(auto &[eval, m] : bestmoves) {
        make_move(m);
        auto [score, bestmove] = alpha_beta(-DBL_MAX, -alpha, d, ab_store, true); score = -score;
        eval = score;
        if(eval >= alpha) {
          alpha = eval;
        }
        retract_move();
      }
      std::sort(bestmoves.begin(), bestmoves.end());
      std::reverse(bestmoves.begin(), bestmoves.end());
    }
    if(bestmoves.empty())return {DBL_MAX, board::nomove};
    return bestmoves.back();
  }

  size_t nodes_searched = 0;
  double evaluation = DBL_MAX;
  COLOR play_as = WHITE;
  const double UNINITIALIZED = -1e9;
  move_t get_fixed_depth_move(pos_t depth=1) {
    nodes_searched = 0;
    zb_hit = 0, zb_miss = 0;
    play_as = activePlayer();
    std::array<ab_info, ZOBRIST_SIZE> *ab_store = new std::array<ab_info, ZOBRIST_SIZE>{};
    for(size_t i = 0; i < ZOBRIST_SIZE; ++i) {
      ab_store->at(i).info.unset();
    }
//    auto [_, m] = alpha_beta(-1e9, 1e9, depth, *ab_store, true);
    auto [_, m] = iterative_deepening_dfs(depth, *ab_store);
    evaluation = _;
    delete ab_store;
    return m;
  }

  struct perft_info {
    board_info info;
    pos_t depth;
    size_t nodes;
  };
  size_t zb_hit = 0, zb_miss = 0;
  size_t _perft(pos_t depth, std::array<perft_info, ZOBRIST_SIZE> &perft_store) {
    if(depth == 1 || depth == 0) {
      return count_moves();
    }
    zobrist::key_t k = zb_hash();
    board_info info = get_board_info();
    if(perft_store[k].info == info && perft_store[k].depth == depth) {
      ++zb_hit;
      return perft_store[k].nodes;
    }
    bool overwrite = true;
    ++zb_miss;
    size_t nodes = 0;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      make_move(i, j);
      ++nodes_searched;
      nodes += _perft(depth - 1, perft_store);
      retract_move();
    });
    if(overwrite) {
      perft_store[k] = {
        .info = info,
        .depth = depth,
        .nodes = nodes
      };
    }
    return nodes;
  }

  inline size_t perft(pos_t depth=1) {
    auto *perft_store = new std::array<perft_info, ZOBRIST_SIZE>{};
    zb_hit = 0, zb_miss = 0;
    for(zobrist::key_t i = 0; i < perft_store->size(); ++i) {
      perft_store->at(i).info.active_player = NEUTRAL;
    }
    size_t nds = _perft(depth, *perft_store);
    delete perft_store;
    return nds;
  }
};
