#pragma once


#include <Piece.hpp>
#include <MoveLine.hpp>


// kind of like lock_scope, but for moves
template <typename BOARD>
struct MoveScope {
  BOARD &b;
  bool is_not_a_copy = true;

  // make a move on constructor
  INLINE explicit MoveScope(BOARD &b, move_t m) noexcept:
    b(b)
  {
    b.make_move(m);
  }

  INLINE explicit MoveScope(MoveScope &&other):
    b(other.b)
  {
    other.is_not_a_copy = false;
  }

  INLINE explicit MoveScope(MoveScope &other):
    b(other.b)
  {
    other.is_not_a_copy = false;
  }

  // unmake a move when leaving scope
  INLINE ~MoveScope() noexcept {
    if(!is_not_a_copy)return;
    b.retract_move();
  }
};


template <typename BOARD>
INLINE MoveScope<BOARD> make_move_scope(BOARD &b, move_t m) {
  return MoveScope<BOARD>(b, m);
}

template <typename BOARD>
struct RecursiveMoveScope {
  BOARD &b;
  int counter = 0;
  bool is_not_a_copy = true;

  INLINE explicit RecursiveMoveScope(BOARD &b) noexcept:
    b(b)
  {}

  INLINE explicit RecursiveMoveScope(RecursiveMoveScope &&other):
    b(other.b), counter(other.counter)
  {
    other.is_not_a_copy = false;
  }

  INLINE explicit RecursiveMoveScope(RecursiveMoveScope &other):
    b(other.b)
  {
    other.is_not_a_copy = false;
  }

  INLINE void scope(move_t m) {
    b.make_move(m);
    ++counter;
  }

  INLINE ~RecursiveMoveScope() {
    if(!is_not_a_copy)return;
    for(int i = 0; i < counter; ++i) {
      b.retract_move();
    }
  }
};

template <typename BOARD>
INLINE RecursiveMoveScope<BOARD> make_recursive_move_scope(BOARD &b) {
  return RecursiveMoveScope<BOARD>(b);
}


template <typename BOARD>
struct MoveLineScope {
  BOARD &b;
  MoveLine &mline;
  bool is_not_a_copy = true;

  INLINE explicit MoveLineScope(BOARD &b, move_t m, MoveLine &mline):
    b(b), mline(mline)
  {
    b.make_move(m);
    mline.premove(m);
  }

  INLINE explicit MoveLineScope(MoveLineScope &&other):
    b(other.b), mline(other.mline)
  {
    other.is_not_a_copy = false;
  }

  INLINE explicit MoveLineScope(MoveLineScope &other):
    b(other.b), mline(other.mline)
  {
    other.is_not_a_copy = false;
  }

  INLINE ~MoveLineScope() {
    if(!is_not_a_copy)return;
    mline.recall();
    b.retract_move();
  }
};

template <typename BOARD>
INLINE MoveLineScope<BOARD> make_mline_scope(BOARD &b, move_t m, MoveLine &mline) {
  return MoveLineScope<BOARD>(b, m, mline);
}


template <typename BOARD>
struct RecursiveMoveLineScope {
  BOARD &b;
  MoveLine &mline;
  int counter = 0;
  bool is_not_a_copy = true;

  INLINE explicit RecursiveMoveLineScope(BOARD &b, MoveLine &mline):
    b(b), mline(mline)
  {}

  INLINE explicit RecursiveMoveLineScope(RecursiveMoveLineScope &&other):
    b(other.b), mline(other.mline), counter(other.counter)
  {
    other.is_not_a_copy = false;
  }

  INLINE explicit RecursiveMoveLineScope(RecursiveMoveLineScope &other):
    b(other.b), mline(other.mline), counter(other.counter)
  {
    other.is_not_a_copy = false;
  }

  INLINE void scope(move_t m) {
    b.make_move(m);
    mline.premove(m);
    ++counter;
  }

  INLINE ~RecursiveMoveLineScope() {
    if(!is_not_a_copy)return;
    for(int i = 0; i < counter; ++i) {
      mline.recall();
      b.retract_move();
    }
  }
};

template <typename BOARD>
INLINE RecursiveMoveLineScope<BOARD> make_recursive_mline_scope(BOARD &b, MoveLine &mline) {
  return RecursiveMoveLineScope<BOARD>(b, mline);
}
