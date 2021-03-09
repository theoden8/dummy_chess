#pragma once


#include <ctime>
#include <cstdlib>
#include <cstdint>

#include <array>
#include <unordered_set>
#include <vector>

#include <Constants.hpp>
#include <Optimizations.hpp>
#include <Piece.hpp>


#ifndef ZOBRIST_SIZE
#define ZOBRIST_SIZE (1ULL << 22)
#endif

namespace zobrist {

using key_t = uint32_t;
using ind_t = uint16_t;

constexpr ind_t rnd_start_piecepos = 0;
constexpr ind_t rnd_start_enpassant = rnd_start_piecepos + board::SIZE * (board::NO_PIECE_INDICES - 1);
constexpr ind_t rnd_start_castlings = rnd_start_enpassant + 8;
constexpr ind_t rnd_start_moveside = rnd_start_castlings + 4;
constexpr ind_t rnd_size = rnd_start_moveside + 1;
std::array<key_t, rnd_size> rnd_hashes;

// http://vigna.di.unimi.it/ftp/papers/xorshift.pdf
uint64_t seed = 0x78473ULL;

INLINE void set_seed(uint64_t new_seed) {
  seed = new_seed;
}

uint64_t randint() {
  seed ^= seed >> 12;
  seed ^= seed << 25;
  seed ^= seed >> 27;
  return seed * 2685821657736338717ULL;
}

void init() {
  static_assert(rnd_start_moveside + 1 < ZOBRIST_SIZE);
  std::unordered_set<key_t> rnd_seen;
  for(ind_t i = 0; i < rnd_hashes.size(); ++i) {
    key_t r;
    do {
      r = randint() % ZOBRIST_SIZE;
      rnd_hashes[i] = r;
    } while(rnd_seen.find(r) != rnd_seen.end());
    rnd_seen.insert(r);
  }
}


template <typename T> using hash_table = std::array<T, ZOBRIST_SIZE>;
template <typename T> using hash_table_ptr = hash_table<T> *;

template<typename InnerObject>
struct StoreScope {
  hash_table_ptr<InnerObject> &zb_store;
  bool is_outer_scope;

  StoreScope(hash_table_ptr<InnerObject> &scope_ptr):
    zb_store(scope_ptr),
    is_outer_scope(scope_ptr == nullptr)
  {
    if(zb_store == nullptr) {
      zb_store = new hash_table<InnerObject>{};
      for(size_t i = 0; i < ZOBRIST_SIZE; ++i) {
        zb_store->at(i).info.unset();
      }
    }
  }

  hash_table<InnerObject> &get_object() {
    return *zb_store;
  }

  void end_scope() {
    if(is_outer_scope) {
      delete zb_store;
      is_outer_scope = false;
    }
  }

  inline ~StoreScope() {
    end_scope();
  }
};

template <typename InnerObject>
decltype(auto) make_store_object_scope(hash_table_ptr<InnerObject> &zb_store) {
  return StoreScope<InnerObject>(zb_store);
}

} // zobrist
