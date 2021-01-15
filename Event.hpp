#pragma once


#include <Bitboard.hpp>
#include <Constants.hpp>


typedef uint64_t event_t;

namespace event {
  constexpr pos_t killnothing = 0xFF;
  constexpr pos_t enpassantnotrace = 0xFF;

  constexpr pos_t BASIC_MARKER = 0xFF;
  constexpr inline event_t basic(move_t m, pos_t killwhat, pos_t halfmoves, pos_t enpassant=enpassantnotrace, pos_t enpassant_trace=enpassantnotrace) {
    event_t e = 0x00;
    e = (e << 8)  | enpassant_trace;
    e = (e << 8)  | enpassant;
    e = (e << 8)  | halfmoves;
    e = (e << 8)  | killwhat;
    e = (e << 16) | m;
    e = (e << 8)  | BASIC_MARKER;
    return e;
  }

  constexpr pos_t CASTLING_MARKER = 0xFE;
  constexpr inline event_t castling(move_t kingmove, move_t rookmove, pos_t halfmoves, pos_t enpassant) {
    event_t e = 0x00;
    e = (e << 8)  | enpassant;
    e = (e << 8)  | halfmoves;
    e = (e << 16) | rookmove;
    e = (e << 16) | kingmove;
    e = (e << 8)  | CASTLING_MARKER;
    return e;
  }

  constexpr pos_t ENPASSANT_MARKER = 0xFD;
  constexpr inline event_t enpassant(move_t m, pos_t killwhere, pos_t halfmoves, pos_t enpassant) {
    event_t e = 0x00;
    e = (e << 8)  | enpassant;
    e = (e << 8)  | halfmoves;
    e = (e << 8)  | killwhere;
    e = (e << 16) | m;
    e = (e << 8)  | ENPASSANT_MARKER;
    return e;
  }

  constexpr pos_t PROMOTION_MARKER = 0xFC;
  // set j in the basic event to have the promotion flag
  constexpr inline event_t promotion_from_basic(event_t basicevent) {
    return ((basicevent >> 8) << 8) | PROMOTION_MARKER;
  }

  inline pos_t extract_byte(event_t &ev) {
    pos_t byte = ev & 0xFF;
    ev >>= 8;
    return byte;
  }
} // namespace event
