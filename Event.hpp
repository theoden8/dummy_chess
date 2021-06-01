#pragma once


#include <Bitmask.hpp>
#include <Bitboard.hpp>


typedef uint_fast32_t event_t;

namespace event {
  constexpr event_t noevent = 0x00;
  constexpr event_t NULLMOVE_MARKER = 0xFF;

  constexpr pos_t marker_bits = 2;
  constexpr pos_t piece_ind_bits = 4;
  constexpr pos_t pos_bits = 6;
  constexpr pos_t move_bits = 14;
  constexpr pos_t move_bits_noprom = move_bits;
  constexpr pos_t killnothing = (1 << piece_ind_bits) - 1;
  constexpr pos_t enpassantnotrace = (1 << pos_bits) - 1;

  constexpr uint8_t BASIC_MARKER = 0x00;
  constexpr inline event_t basic(move_t m, pos_t killwhat, pos_t enpassant_trace=enpassantnotrace) {
    event_t e = 0x00;
    e = (e << pos_bits)  | enpassant_trace;
    e = (e << piece_ind_bits) | killwhat;
    e = (e << move_bits) | m;
    e = (e << marker_bits) | BASIC_MARKER;
    return e;
  }

  constexpr uint8_t CASTLING_MARKER = 0x01;
  constexpr inline event_t castling(move_t kingmove, move_t rookmove) {
    event_t e = 0x00;
    e = (e << move_bits) | rookmove;
    e = (e << move_bits) | kingmove;
    e = (e << marker_bits) | CASTLING_MARKER;
    return e;
  }

  constexpr uint8_t ENPASSANT_MARKER = 0x02;
  constexpr inline event_t enpassant(move_t m, pos_t killwhere) {
    event_t e = 0x00;
    e = (e << pos_bits)  | killwhere;
    e = (e << move_bits) | m;
    e = (e << marker_bits) | ENPASSANT_MARKER;
    return e;
  }

  constexpr uint8_t PROMOTION_MARKER = 0x03;
  // set j in the basic event has the promotion flag
  constexpr inline event_t promotion_from_basic(event_t basicevent) {
    return ((basicevent >> marker_bits) << marker_bits) | PROMOTION_MARKER;
  }

  inline uint8_t extract_marker(event_t &ev) {
    if(ev == event::noevent) {
      return NULLMOVE_MARKER;
    }
    pos_t byte = ev & ((1 << marker_bits) - 1);
    ev >>= marker_bits;
    return byte;
  }

  inline pos_t extract_byte(event_t &ev) {
    pos_t byte = ev & 0xFF;
    ev >>= 8;
    return byte;
  }

  inline pos_t extract_piece_ind(event_t &ev) {
    pos_t p = ev & ((1 << piece_ind_bits) - 1);
    ev >>= piece_ind_bits;
    return p;
  }

  inline pos_t extract_pos(event_t &ev) {
    pos_t p = ev & ((1 << pos_bits) - 1);
    ev >>= 6;
    return p;
  }

  inline move_t extract_move(event_t &ev) {
    move_t m = ev & ((1ULL << move_bits) - 1);
    ev >>= move_bits;
    return m;
  }

  inline move_t extract_move_noprom(event_t &ev) {
    move_t m = ev & ((1ULL << move_bits_noprom) - 1);
    ev >>= move_bits_noprom;
    return m;
  }
} // namespace event
