#pragma once


#ifdef FLAG_JEMALLOC_EXTERNAL
#include <jemalloc/jemalloc.h>
#endif

#include <ctime>
#include <cstdlib>
#include <cstdint>

#ifdef FLAG_BSD
#include <bsd/stdlib.h>
#endif

#include <memory>
#include <array>
#include <unordered_set>
#include <vector>

#include <Optimizations.hpp>
#include <Piece.hpp>


#ifndef ZOBRIST_SIZE
#define ZOBRIST_SIZE (1ULL << 19)
#endif

namespace zobrist {

using key_t = uint_fast32_t;
using ind_t = uint_fast16_t;

constexpr ind_t rnd_start_piecepos = 0;
constexpr ind_t rnd_start_enpassant = rnd_start_piecepos + ind_t(board::SIZE) * ind_t(board::NO_PIECE_INDICES);
constexpr ind_t rnd_start_castlings = rnd_start_enpassant + 8;
constexpr ind_t rnd_start_moveside = rnd_start_castlings + 4;
constexpr ind_t rnd_size = rnd_start_moveside + 1;
std::array<key_t, rnd_size> rnd_hashes;

INLINE zobrist::key_t piece_hash(COLOR c, PIECE p, pos_t square) {
  return rnd_hashes[rnd_start_piecepos + ind_t(board::SIZE) * Piece(p,c).piece_index + square];
}

// http://vigna.di.unimi.it/ftp/papers/xorshift.pdf
uint64_t g_seed = 0x78473ULL;

INLINE void set_seed(uint64_t new_seed) {
  g_seed = new_seed;
}

INLINE uint64_t randint() {
  g_seed ^= g_seed >> 12;
  g_seed ^= g_seed << 25;
  g_seed ^= g_seed >> 27;
  return g_seed * 2685821657736338717ULL;
}

void init(size_t zbsize) {
#if defined(NDEBUG) && !defined(FLAG_PROFILING)
  set_seed(((uint64_t)arc4random() << 32) | arc4random());
#endif
  assert(bitmask::is_exp2(zbsize));
  assert(rnd_start_moveside + 1 < zbsize);
  std::unordered_set<key_t> rnd_seen = {0};
  for(ind_t i = 0; i < rnd_hashes.size(); ++i) {
    key_t r;
    do {
      r = randint() & (zbsize - 1);
      rnd_hashes[i] = r;
    } while(rnd_seen.find(r) != rnd_seen.end());
    rnd_seen.insert(r);
  }
}


INLINE void toggle_pos(key_t &zb, pos_t piece_index, pos_t pos) {
  assert(piece_index < board::NO_PIECE_INDICES);
  zb ^= zobrist::rnd_hashes[zobrist::rnd_start_piecepos + zobrist::ind_t(board::SIZE) * piece_index + pos];
}

INLINE void move_pos_quiet(key_t &zb, pos_t piece_index, pos_t i, pos_t j) {
  zobrist::toggle_pos(zb, piece_index, i);
  zobrist::toggle_pos(zb, piece_index, j);
}

INLINE void move_pos_capture(key_t &zb, pos_t piece_index, pos_t i, pos_t piece_index_victim, pos_t j) {
  assert(piece_index_victim < board::NO_PIECE_INDICES);
  zobrist::toggle_pos(zb, piece_index_victim, j);
  zobrist::move_pos_quiet(zb, piece_index, i, j);
}

INLINE void toggle_castling(key_t &zb, COLOR c, CASTLING_SIDE side) {
  zb ^= zobrist::rnd_hashes[zobrist::rnd_start_castlings + board::_castling_index(c, side)];
}


template <typename T> using ttable = std::vector<T>;

template<typename InnerObject>
struct StoreScope {
  std::shared_ptr<zobrist::ttable<InnerObject>> &zb_store;

  explicit INLINE StoreScope(std::shared_ptr<zobrist::ttable<InnerObject>> &scope_ptr, size_t zbsize):
    zb_store(scope_ptr)
  {
    if(!zb_store) {
      zb_store.reset(new zobrist::ttable<InnerObject>(zbsize));
      reset();
    }
  }

  explicit StoreScope(const StoreScope<InnerObject> &other) = delete;
  explicit INLINE StoreScope(StoreScope<InnerObject> &&other):
    zb_store(other.zb_store)
  {}

  void reset() {
    for(size_t i = 0; i < zb_store->size(); ++i) {
      zb_store->at(i).info.unset();
    }
  }

  zobrist::ttable<InnerObject> &get_object() {
    return *zb_store;
  }
};

template <typename InnerObject>
INLINE decltype(auto) make_store_object_scope(std::shared_ptr<zobrist::ttable<InnerObject>> &zb_store, size_t zbsize) {
  return StoreScope<InnerObject>(zb_store, zbsize);
}

template <typename BoardT>
zobrist::key_t zb_hash(const BoardT &board) {
  zobrist::key_t zb = 0x00;
  for(piece_index_t i = 0; i < board::NO_PIECE_INDICES; ++i) {
    const Piece p = BoardT::pieces[i];
    bitmask::foreach(board.get_mask(p), [&](pos_t pos) mutable -> void {
      zb ^= zobrist::rnd_hashes[zobrist::rnd_start_piecepos + board::SIZE * i + pos];
    });
  }
  if(board.is_castling(WHITE,QUEEN_SIDE))zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 0];
  if(board.is_castling(WHITE,KING_SIDE)) zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 1];
  if(board.is_castling(BLACK,QUEEN_SIDE))zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 2];
  if(board.is_castling(BLACK,KING_SIDE)) zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 3];
  if(board.enpassant_trace() != board::nopos) {
    zb ^= zobrist::rnd_hashes[zobrist::rnd_start_enpassant + board::_x(board.enpassant_trace())];
  }
  if(board.activePlayer() == BLACK) {
    zb ^= zobrist::rnd_hashes[zobrist::rnd_start_moveside];
  }
  return zb;
}

template <typename BoardT>
zobrist::key_t zb_hash_material(const BoardT &board, bool mirror) {
  zobrist::key_t zb = 0x00;
  for(piece_index_t i = 0; i < board::NO_PIECE_INDICES; ++i) {
    const Piece p = BoardT::pieces[i];
    pos_t pcount = piece::size(board.get_mask(p));
    if(board.crazyhouse)pcount += board.n_subs[i];
    for(pos_t x = 0; x < pcount; ++x) {
      zb ^= zobrist::piece_hash(!mirror ? p.color : enemy_of(p.color), p.value, x);
    }
  }
  return zb;
}

} // zobrist
