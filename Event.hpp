#pragma once


#include <Bitboard.hpp>
#include <Constants.hpp>


typedef uint64_t event_t;

namespace event {
  constexpr pos_t killnothing = 0xFF;
  constexpr pos_t enpassantnotrace = 0xFF;

  constexpr inline pos_t compress_castlings(piece_bitboard_t castlings) {
    pos_t comp = 0x00;
    comp |= (castlings & 0x04LLU) ? 1 : 0; comp <<= 1;
    comp |= (castlings & 0x40LLU) ? 1 : 0; comp <<= 1;
    comp |= (castlings & (0x04LLU << (board::SIZE-board::LEN))) ? 1 : 0; comp <<= 1;
    comp |= (castlings & (0x40LLU << (board::SIZE-board::LEN))) ? 1 : 0; comp <<= 1;
    return comp;
  }

  constexpr inline piece_bitboard_t decompress_castlings(pos_t comp) {
    piece_bitboard_t castlings = 0x00;
    if(comp&1)castlings|=(0x40LLU << (board::SIZE-board::LEN));
    comp>>=1;
    if(comp&1)castlings|=(0x04LLU << (board::SIZE-board::LEN));
    comp>>=1;
    if(comp&1)castlings|=(0x40LLU);
    comp>>=1;
    if(comp&1)castlings|=(0x04LLU);
    comp>>=1;
    return castlings;
  }

  constexpr pos_t BASIC_MARKER = 0xFF;
  constexpr inline event_t basic(pos_t from, pos_t to, pos_t killwhat=killnothing, piece_bitboard_t castlings=0xff,
                                 pos_t enpassant_trace=enpassantnotrace, pos_t enpassant_layman=0)
  {
    event_t e = 0x00;
    e = (e << 8) | enpassant_layman;
    e = (e << 8) | enpassant_trace;
    e = (e << 8) | compress_castlings(castlings);
    e = (e << 8) | killwhat;
    e = (e << 8) | to;
    e = (e << 8) | from;
    e = (e << 8) | BASIC_MARKER;
    return e;
  }

  constexpr pos_t CASTLING_MARKER = 0xFE;
  constexpr inline event_t castling(pos_t kingfrom, pos_t kingto, pos_t rookfrom, pos_t rookto, piece_bitboard_t castlings,
                                    pos_t enpassant_trace=enpassantnotrace, pos_t enpassant_layman=0)
  {
    event_t e = 0x00;
    e = (e << 8) | enpassant_layman;
    e = (e << 8) | enpassant_trace;
    e = (e << 8) | compress_castlings(castlings);
    e = (e << 8) | rookto;
    e = (e << 8) | rookfrom;
    e = (e << 8) | kingto;
    e = (e << 8) | kingfrom;
    e = (e << 8) | CASTLING_MARKER;
    return e;
  }

  constexpr pos_t ENPASSANT_MARKER = 0xFD;
  constexpr inline event_t enpassant(pos_t from, pos_t to, pos_t killwhere, piece_bitboard_t castlings,
                                     pos_t enpassant_trace=enpassantnotrace, pos_t enpassant_layman=0)
  {
    event_t e = 0x00;
    e = (e << 8) | enpassant_layman;
    e = (e << 8) | enpassant_trace;
    e = (e << 8) | compress_castlings(castlings);
    e = (e << 8) | killwhere;
    e = (e << 8) | to;
    e = (e << 8) | from;
    e = (e << 8) | ENPASSANT_MARKER;
    return e;
  }

  constexpr pos_t PROMOTION_MARKER = 0xFC;
  constexpr inline event_t promotion_from_basic(event_t basicevent, pos_t becomewhat) {
    event_t e = 0x00;
    e = (e << 8) | becomewhat;
    e = (e << (8*6)) | (basicevent >> 8);
    e = (e << 8) | PROMOTION_MARKER;
    return e;
  }

  constexpr inline event_t promotion(pos_t from, pos_t to, pos_t killwhat, pos_t becomewhat, piece_bitboard_t castlings) {
    return promotion_from_basic(event::basic(from, to, killwhat, castlings), becomewhat);
  }

  pos_t extract_byte(event_t &ev) {
    pos_t byte = ev & 0xff;
    ev >>= 8;
    return byte;
  }

  piece_bitboard_t extract_castlings(event_t &ev) {
    return decompress_castlings(extract_byte(ev));
  }
} // namespace event
