#pragma once


#include <cfloat>
#include <algorithm>
#include <list>
#include <map>

#include <FEN.hpp>
#include <Board.hpp>


class Engine : public Board {
public:
  Engine(const fen::FEN fen=fen::starting_pos):
    Board(fen)
  {}

  static constexpr std::array<pos_t, 4> PROMOTION_PIECES = {
    board::PROMOTE_KNIGHT, board::PROMOTE_BISHOP,
    board::PROMOTE_ROOK, board::PROMOTE_QUEEN
  };

  template <typename F>
  INLINE void iter_moves_from(pos_t i, F &&func) const {
    bitmask::foreach(get_moves_from(i), [&](pos_t j) mutable -> void {
      if(is_promotion_move(i, j)) {
        for(pos_t promotion : PROMOTION_PIECES) {
          func(i, j | promotion);
        }
      } else {
        func(i, j);
      }
    });
  }

  template <typename F>
  INLINE void iter_capture_moves_from(pos_t i, F &&func) const {
    const COLOR c = self[i].color;
    const piece_bitboard_t foes = state_piece_positions[enemy_of(c)];
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      if(is_promotion_move(i, j)
          || is_enpassant_take_move(i, j)
          || self[j & board::MOVEMASK].color == enemy_of(c))
      {
        func(i, j);
      }
    });
  }

  piece_bitboard_t get_capture_moves_from(pos_t i) const {
    piece_bitboard_t pb = 0x00;
    iter_capture_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      pb |= 1ULL << (j & board::MOVEMASK);
    });
    return pb;
  }

  template <typename F>
  ALWAYS_UNROLL INLINE void iter_moves(F &&func) const {
    bitmask::foreach(state_piece_positions[activePlayer()], [&](pos_t i) mutable -> void {
      iter_moves_from(i, func);
    });
  }

  template <typename F>
  ALWAYS_UNROLL INLINE void iter_capture_moves(F &&func) const {
    bitmask::foreach(state_piece_positions[activePlayer()], [&](pos_t i) mutable -> void {
      iter_capture_moves_from(i, func);
    });
  }

  INLINE size_t count_moves() const {
    int16_t no_moves = 0;
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
    bitmask::foreach(state_piece_positions[c], [&](pos_t i) mutable -> void {
      auto a = self.get_attacks_from(i);
      attacks += bitmask::count_bits(a & occupied) + bitmask::count_bits(a);
    });
    return attacks;
  }

  double heuristic_of(COLOR c) const {
    double h = 0;
    if(self.is_draw())return 0;
    // checkmate
    if(!can_move())return -1e7;

    h += h_material(c);
    h += h_pins(c) * 1e-2;
    h += h_attack_cells(c) * 1e-4;
    return h;
  }

  double evaluate() const {
    double score = heuristic_of(WHITE) - heuristic_of(BLACK);
    return play_as==WHITE ? score : -score;
  }

  double move_heuristic(pos_t i, pos_t j) const {
    double val = material_of(self[j & board::MOVEMASK].value);
    if(is_enpassant_take_move(i, j)) {
      val += material_of(PAWN);
    }
    if(is_promotion_move(i, j)) {
      val += material_of(board::get_promotion_as(j)) - material_of(PAWN);
    }
    if(self[j & board::MOVEMASK].value != EMPTY) {
      val -= material_of(self[i].value)*.05;
    }
    return val;
  }

  struct ab_info {
    board_info info;
    int16_t depth;
    double eval;
    move_t m;
    MoveLine subpline;
  };
  double alpha_beta_quiscence(double alpha, double beta, int16_t depth, MoveLine &pline,
                              std::array<ab_info, ZOBRIST_SIZE> &ab_store)
  {
//    str::print("abq: ", _line_str(pline.full()));
    double score = evaluate();
    if(score >= beta) {
//      str::print("score good", score, ">=", beta);
      ++nodes_searched;
      return beta;
    } else if(score > alpha) {
      alpha = score;
    }

    zobrist::key_t k = zb_hash();
    auto info = get_board_info();
    if(ab_store[k].info == info && depth <= ab_store[k].depth) {
      ++zb_hit;
      ++nodes_searched;
      pline.replace_line(ab_store[k].subpline);
      return ab_store[k].eval;
    }
    bool overwrite = (depth + 1 >= ab_store[k].depth);

    std::vector<std::pair<double, move_t>> capturemoves;
    capturemoves.reserve(8);
    iter_capture_moves([&](pos_t i, pos_t j) mutable -> void {
      double val = move_heuristic(i, j);
      const move_t m = bitmask::_pos_pair(i, j);
      //assert(check_valid_move(m));
//      if(m == pline.front()) {
//        val += 100;
//      } else if(std::find(pline.begin(), pline.end(), m) != std::end(pline)) {
//        val += 10;
//      }
//      str::print("capturemove", _move_str(m));
      capturemoves.emplace_back(-val, m);
    });
    if(capturemoves.empty()) {
//      str::print("no moves, score:", score);
      return score;
    } else {
//      str::print("capturemoves length", capturemoves.size());
    }
    std::sort(capturemoves.begin(), capturemoves.end());
    assert(pline.empty());
    for(const auto [_, m] : capturemoves) {
      //assert(check_valid_move(m));
      volatile auto guard = move_guard(m);
      MoveLine pline_alt = pline.branch_from_past();
      pline_alt.premove(m);
      score = -alpha_beta_quiscence(-beta, -alpha, depth - 1, pline_alt, ab_store);
      if(!check_valid_sequence(pline_alt)) {
        str::print("pline not playable", _line_str(pline_alt.full()));
        str::print("sequence not valid:", _line_str(pline_alt));
        str::print("FEN: ", fen::export_as_string(self.export_as_fen()));
      }
      pline_alt.recall();
      if(score >= beta) {
        pline.replace_line(pline_alt);
//        str::print("new cut-off pline", _line_str(pline.full()));
        if(overwrite) {
          ab_store[k] = { .info=info, .depth=depth, .eval=beta, .subpline=pline_alt };
        }
        return beta;
      } else if(score > alpha) {
        pline.replace_line(pline_alt);
//        str::print("new pline", _line_str(pline.full()));
        alpha = score;
      }
    }
    if(overwrite) {
      ab_store[k] = { .info=info, .depth=depth, .eval=alpha, .subpline=pline };
    }
    ++nodes_searched;
    return alpha;
  }

  std::pair<double, move_t> alpha_beta(double alpha, double beta, int16_t depth, MoveLine &pline,
                                       std::array<ab_info, ZOBRIST_SIZE> &ab_store)
  {
    if(depth == 0) {
      const double score = alpha_beta_quiscence(alpha, beta, depth, pline, ab_store);
      //const double score = evaluate();
      return {score, board::nomove};
    }
    zobrist::key_t k = zb_hash();
    auto info = get_board_info();
    if(ab_store[k].info == info && depth <= ab_store[k].depth) {
      ++zb_hit;
      pline.replace_line(ab_store[k].subpline);
      return {ab_store[k].eval, ab_store[k].m};
    }
    bool overwrite = (depth + 1 >= ab_store[k].depth);
    ++zb_miss;
    std::vector<std::pair<double, move_t>> moves;
    moves.reserve(16);
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      // move-ordering heuristic
      double val = move_heuristic(i, j);
      const move_t m = bitmask::_pos_pair(i, j);
      // principal variation move ordering
//      if(m == pline.front()) {
//        val += 100;
//      } else if(std::find(pline.begin(), pline.end(), m) != std::end(pline)) {
//        val += 10;
//      }
      moves.emplace_back(-val, bitmask::_pos_pair(i,j));
    });
    if(moves.empty()) {
      return {evaluate(), board::nomove};
    }
    std::sort(moves.begin(), moves.end());
    move_t m_best = board::nomove;
    assert(pline.empty());
    for(const auto [_, m] : moves) {
      volatile auto guard = move_guard(m);
      MoveLine pline_alt = pline.branch_from_past();
      pline_alt.premove(m);
      const double score = -alpha_beta(-beta, -alpha, depth - 1, pline_alt, ab_store).first;
      if(!check_valid_sequence(pline_alt)) {
        str::print("pline not playable", _line_str(pline_alt.full()));
        str::print("sequence not valid:", _line_str(pline_alt));
        str::print("FEN: ", fen::export_as_string(self.export_as_fen()));
      }
      pline_alt.recall();
      //printf("depth=%d, score=%.5f, %s\n", depth,score,board::_move_str(m).c_str());
      if(score >= beta) {
        pline.replace_line(pline_alt);
        if(overwrite) {
          ab_store[k] = { .info=info, .depth=depth, .eval=beta, .m=m, .subpline=pline };
        }
        return {beta, m};
      } else if(score > alpha) {
        pline.replace_line(pline_alt);
        alpha = score;
        m_best = m;
      }
    };
    if(overwrite) {
      ab_store[k] = { .info=info, .depth=depth, .eval=alpha, .m=m_best, .subpline=pline };
    }
//    str::print("depth=", depth, "new best move", _line_str(pline));
    return {alpha, m_best};
  }

  bool play_as = WHITE;
  size_t nodes_searched = 0;
  double evaluation = DBL_MAX;
  const double UNINITIALIZED = -1e9;
  template <typename F>
  move_t get_fixed_depth_move(int depth, F &&callback_f, const std::unordered_set<move_t> &searchmoves) {
    play_as = activePlayer();
    nodes_searched = 0;
    zb_hit = 0, zb_miss = 0;
    std::array<ab_info, ZOBRIST_SIZE> *ab_store = new std::array<ab_info, ZOBRIST_SIZE>{};
    for(size_t i = 0; i < ZOBRIST_SIZE; ++i) {
      ab_store->at(i).info.unset();
    }
    MoveLine pline;
    auto [_, m] = alpha_beta(-1e9, 1e9, depth, pline, *ab_store);
    str::print("pvline:", _line_str(pline), "size:", pline.size());
    evaluation = _;
    delete ab_store;
    if(!check_valid_sequence(pline)) {
      str::print("pvline not playable");
    }
    return m;
  }

  decltype(auto) get_fixed_depth_move(int depth, const std::unordered_set<move_t> &searchmoves={}) {
    return get_fixed_depth_move(depth, [](const move_t m, const MoveLine &pline) mutable -> bool {return true;}, searchmoves);
  }
//
  struct perft_info {
    board_info info;
    int depth;
    size_t nodes;
  };
  size_t zb_hit = 0, zb_miss = 0;
  size_t _perft(int depth, std::array<perft_info, ZOBRIST_SIZE> &perft_store) {
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
      volatile auto guard = move_guard(bitmask::_pos_pair(i, j));
      ++nodes_searched;
      nodes += _perft(depth - 1, perft_store);
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

  inline size_t perft(int depth=1) {
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
