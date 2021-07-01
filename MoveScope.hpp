#pragma once


#include <Bitboard.hpp>
#include <MoveLine.hpp>


// kind of like lock_scope, but for moves
template <typename BOARD>
struct MoveScope {
  BOARD &b;

  // make a move on constructor
  inline explicit MoveScope(BOARD &b, move_t m) noexcept:
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

  inline explicit RecursiveMoveScope(BOARD &b) noexcept:
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


template <typename BOARD>
struct MoveLineScope {
  BOARD &b;
  MoveLine &mline;

  inline explicit MoveLineScope(BOARD &b, move_t m, MoveLine &mline):
    b(b), mline(mline)
  {
    b.make_move(m);
    mline.premove(m);
  }

  inline ~MoveLineScope() {
    mline.recall();
    b.retract_move();
  }
};

template <typename BOARD>
inline MoveLineScope<BOARD> make_mline_scope(BOARD &b, move_t m, MoveLine &mline) {
  return MoveLineScope<BOARD>(b, m, mline);
}


template <typename BOARD>
struct RecursiveMoveLineScope {
  BOARD &b;
  MoveLine &mline;
  int counter = 0;

  inline explicit RecursiveMoveLineScope(BOARD &b, MoveLine &mline):
    b(b), mline(mline)
  {}

  inline void scope(move_t m) {
    b.make_move(m);
    mline.premove(m);
    ++counter;
  }

  inline ~RecursiveMoveLineScope() {
    for(int i = 0; i < counter; ++i) {
      mline.recall();
      b.retract_move();
    }
  }
};

template <typename BOARD>
inline RecursiveMoveLineScope<BOARD> make_recursive_mline_scope(BOARD &b, MoveLine &mline) {
  return RecursiveMoveLineScope<BOARD>(b, mline);
}
