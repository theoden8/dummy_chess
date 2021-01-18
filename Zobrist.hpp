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

void init() {
  srand(time(nullptr));
  static_assert(rnd_start_moveside + 1 < ZOBRIST_SIZE);
  std::unordered_set<key_t> rnd_seen;
  for(ind_t i = 0; i < rnd_hashes.size(); ++i) {
    key_t r;
    do {
      r = rand() % ZOBRIST_SIZE;
      rnd_hashes[i] = r;
    } while(rnd_seen.find(r) != rnd_seen.end());
    rnd_seen.insert(r);
  }
}

} // zobrist
