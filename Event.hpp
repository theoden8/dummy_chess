#pragma once


#include <Bitboard.hpp>


// event sequance that can be reverted
typedef enum { NOEVENT, DEATH, KILL, SPAWN } EVENT;
struct event {
  EVENT type;
  pos_t position;
  constexpr event():
    type(NOEVENT), position(UCHAR_MAX)
  {}
};
