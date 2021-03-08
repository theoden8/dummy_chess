#pragma once


#include <Constants.hpp>


// kind of like lock_scope, but for moves
template <typename BOARD>
struct MoveScope {
  BOARD &b;

  // make a move on constructor
  inline MoveScope(BOARD &b, move_t m) noexcept:
    b(b)
  {
    b.make_move(m);
  }

  // unmake a move when leaving scope
  inline ~MoveScope() noexcept {
    b.retract_move();
  }
};


template <typename BOARD>
inline MoveScope<BOARD> make_move_scope(BOARD &b, move_t m) {
  return MoveScope<BOARD>(b, m);
}

template <typename BOARD>
struct RecursiveMoveScope {
  BOARD &b;
  int counter = 0;

  inline RecursiveMoveScope(BOARD &b) noexcept:
    b(b)
  {}

  inline void scope(move_t m) noexcept {
    b.make_move(m);
    ++counter;
  }

  inline ~RecursiveMoveScope() noexcept {
    for(int i = 0; i < counter; ++i) {
      b.retract_move();
    }
  }
};

template <typename BOARD>
inline RecursiveMoveScope<BOARD> make_recursive_move_scope(BOARD &b) {
  return RecursiveMoveScope<BOARD>(b);
}
