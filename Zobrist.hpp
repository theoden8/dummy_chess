#pragma once


#include <array>
#include <unordered_set>
#include <vector>

#include <bsd/stdlib.h>

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
  static_assert(rnd_start_moveside + 1 < ZOBRIST_SIZE);
  std::unordered_set<key_t> rnd_seen;
  for(ind_t i = 0; i < rnd_hashes.size(); ++i) {
    key_t r;
    do {
      r = arc4random() % ZOBRIST_SIZE;
      rnd_hashes[i] = r;
    } while(rnd_seen.find(r) != rnd_seen.end());
    rnd_seen.insert(r);
  }
}

INLINE key_t zb_hash_piece(piece_bitboard_t mask, PIECE p, COLOR c) {
  key_t zb = 0x00;
  bitmask::foreach(mask, [&](pos_t pos) mutable -> void {
    zb ^= rnd_hashes[rnd_start_piecepos + board::SIZE * Piece::get_piece_index(p,c) + pos];
  });
  return zb;
}

INLINE key_t zb_hash_castlings(std::vector<bool> castlings) {
  key_t zb = 0x00;
  if(castlings[0])zb^=rnd_hashes[rnd_start_castlings + 0];
  if(castlings[1])zb^=rnd_hashes[rnd_start_castlings + 1];
  if(castlings[2])zb^=rnd_hashes[rnd_start_castlings + 2];
  if(castlings[3])zb^=rnd_hashes[rnd_start_castlings + 3];
  return zb;
}

INLINE key_t zb_hash_enpassant(pos_t enpassant) {
  if(enpassant == event::enpassantnotrace) {
    return 0x00;
  }
  return rnd_hashes[rnd_start_enpassant + board::_x(enpassant)];
}

INLINE key_t zb_hash_player(COLOR c) {
  return (c == BLACK) ? rnd_hashes[rnd_start_moveside] : 0x00ULL;
}

} // zobrist
