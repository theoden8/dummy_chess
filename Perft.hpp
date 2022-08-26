#pragma once

#include <memory>

#include <Board.hpp>


class Perft : public Board {
public:
  using depth_t = int16_t;

  static constexpr std::array<pos_t, 4> PROMOTION_PIECES = {
    board::PROMOTE_KNIGHT, board::PROMOTE_BISHOP,
    board::PROMOTE_ROOK, board::PROMOTE_QUEEN
  };

  template <typename F>
  INLINE void iter_moves_from(pos_t i, F &&func) const {
    bitmask::foreach(state.moves[i], [&](pos_t j) mutable -> void {
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
  INLINE void iter_drop_moves(F &&func) const {
    if(crazyhouse) {
      if(is_draw_halfmoves()||is_draw_material())return;
      const COLOR c = activePlayer();
      const piece_bitboard_t ch_drop_locations = ~(bits[WHITE]|bits[BLACK]) & state.checkline[c];
      {
        if(n_subs[Piece::get_piece_index(PAWN, c)]) {
          bitmask::foreach(ch_drop_locations & board::PAWN_RANKS, [&](pos_t j) mutable -> void {
            func(board::DROP_PAWN, j);
          });
        }
      }
      for(PIECE p : {KNIGHT, BISHOP, ROOK, QUEEN}) {
        if(n_subs[Piece::get_piece_index(p, c)]) {
          bitmask::foreach(ch_drop_locations, [&](pos_t j) mutable -> void {
            func(pos_t(p) | board::CRAZYHOUSE_DROP, j);
          });
        }
      }
    }
  }

  template <typename F>
  INLINE void iter_moves(F &&func) const {
    const COLOR c = activePlayer();
    bitmask::foreach(bits[c], [&](pos_t i) mutable -> void {
      iter_moves_from(i, std::forward<F>(func));
    });
    iter_drop_moves(std::forward<F>(func));
  }

  INLINE size_t count_moves(COLOR c) const {
    assert(c < NO_COLORS);
    uint16_t no_moves = 0;
    bitmask::foreach(bits[c], [&](pos_t i) mutable -> void {
      pos_t moves_from = bitmask::count_bits(state.moves[i]);
      if(piece::is_set(bits_pawns, i) && (board::_y(i) == 2-1 || board::_y(i) == 7-1)
          && (
            (self.color_at_pos(i) == WHITE && 1+board::_y(i) == 7)
            || (self.color_at_pos(i) == BLACK && 1+board::_y(i) == 2))
        )
      {
        moves_from *= 4;
      }
      no_moves += moves_from;
    });
    if(crazyhouse) {
      const COLOR c = activePlayer();
      const piece_bitboard_t ch_drop_locations = ~(bits[WHITE]|bits[BLACK]) & state.checkline[c];
      if(n_subs[Piece::get_piece_index(PAWN, c)]) {
        no_moves += piece::size(ch_drop_locations & board::PAWN_RANKS);
      }
      pos_t npieces = 0;
      for(PIECE p : {KNIGHT, BISHOP, ROOK, QUEEN}) {
        if(n_subs[Piece::get_piece_index(p, c)]) {
          ++npieces;
        }
      }
      no_moves += npieces * piece::size(ch_drop_locations);
    }
    return no_moves;
  }

  // for MC-style testing
  INLINE move_t get_random_move() const {
    std::vector<move_t> moves;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nullmove;
    return moves[rand() % moves.size()];
  }

  INLINE move_t get_random_move_from(pos_t i) const {
    std::vector<move_t> moves;
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nullmove;
    return moves[rand() % moves.size()];
  }

  struct tt_perft_entry {
    board_info info;
    depth_t depth;
    size_t nodes;
  };

  size_t nodes_searched = 0;
  size_t zb_hit = 0, zb_miss = 0, zb_occupied = 0;

  std::shared_ptr<zobrist::ttable<tt_perft_entry>> perft_ttable = nullptr;
  decltype(auto) get_zobrist_perft_scope() {
    const size_t size_perft = zobrist_size;
    const size_t mem_perft = size_perft * sizeof(tt_perft_entry);
    return zobrist::make_store_object_scope<tt_perft_entry>(perft_ttable, size_perft);
  }

  size_t _perft(depth_t depth, std::vector<tt_perft_entry> &perft_ttable) {
    if(depth == 1 || depth == 0) {
      return count_moves(activePlayer());
    }
    // look-up:
    const zobrist::key_t k = zb_hash();
    if(perft_ttable[k].info == state.info && perft_ttable[k].depth == depth) {
      ++zb_hit;
      return perft_ttable[k].nodes;
    } else {
      if(perft_ttable[k].info.is_unset())++zb_occupied;
      ++zb_miss;
    }
    // search
    constexpr bool overwrite = true;
    size_t nodes = 0;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      decltype(auto) mscope = self.move_scope(bitmask::_pos_pair(i, j));
      ++nodes_searched;
      nodes += _perft(depth - 1, perft_ttable);
    });
    if(overwrite) {
      perft_ttable[k] = { .info=state.info, .depth=depth, .nodes=nodes };
    }
    return nodes;
  }

  INLINE size_t perft(depth_t depth=1) {
    decltype(auto) store_scope = get_zobrist_perft_scope();
    zb_hit = 0, zb_miss = 0, zb_occupied = 0;
    return _perft(depth, store_scope.get_object());
  }

  explicit Perft(const fen::FEN &fen=fen::starting_pos, size_t zbsize=ZOBRIST_SIZE):
    Board(fen, zbsize)
  {}
};
