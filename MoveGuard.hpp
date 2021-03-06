#pragma once


#include <Constants.hpp>


// kind of like lock_guard, but for moves
template <typename BOARD>
struct MoveGuard {
  BOARD &b;

  // make a move on constructor
  inline MoveGuard(BOARD &b, move_t m) noexcept:
    b(b)
  {
    b.make_move(m);
  }

  // unmake a move when leaving scope
  inline ~MoveGuard() noexcept {
    b.retract_move();
  }
};


template <typename BOARD>
inline MoveGuard<BOARD> make_move_guard(BOARD &b, move_t m) {
  return MoveGuard<BOARD>(b, m);
}

template <typename BOARD>
struct RecursiveMoveGuard {
  BOARD &b;
  int counter = 0;

  inline RecursiveMoveGuard(BOARD &b) noexcept:
    b(b)
  {}

  inline void guard(move_t m) noexcept {
    b.make_move(m);
    ++counter;
  }

  inline ~RecursiveMoveGuard() noexcept {
    for(int i = 0; i < counter; ++i) {
      b.retract_move();
    }
  }
};

template <typename BOARD>
inline RecursiveMoveGuard<BOARD> make_recursive_move_guard(BOARD &b) {
  return RecursiveMoveGuard<BOARD>(b);
}
