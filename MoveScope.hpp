#pragma once


#include <Piece.hpp>
#include <MoveLine.hpp>


// kind of like lock_scope, but for moves
template <board::IndexableView BoardT>
struct MoveScope {
  BoardT &b;
  bool is_acting_scope = true;

  // make a move on constructor
  INLINE explicit MoveScope(BoardT &b, move_t m) noexcept:
    b(b)
  {
    b.make_move(m);
  }

  explicit MoveScope(const MoveScope &other) = delete;
  explicit INLINE MoveScope(MoveScope &&other):
    b(other.b)
  {
    other.is_acting_scope = false;
  }

  // unmake a move when leaving scope
  INLINE ~MoveScope() noexcept {
    if(!is_acting_scope)return;
    b.retract_move();
  }
};


template <board::IndexableView BoardT>
INLINE decltype(auto) make_move_scope(BoardT &b, move_t m) {
  return MoveScope<BoardT>(b, m);
}

template <board::IndexableView BoardT>
struct RecursiveMoveScope {
  BoardT &b;
  int counter = 0;
  bool is_acting_scope = true;

  INLINE explicit RecursiveMoveScope(BoardT &b) noexcept:
    b(b)
  {}

  explicit RecursiveMoveScope(const RecursiveMoveScope &other) = delete;
  INLINE explicit RecursiveMoveScope(RecursiveMoveScope &&other):
    b(other.b), counter(other.counter)
  {
    other.is_acting_scope = false;
  }

  INLINE void scope(move_t m) {
    b.make_move(m);
    ++counter;
  }

  INLINE ~RecursiveMoveScope() {
    if(!is_acting_scope)return;
    for(int i = 0; i < counter; ++i) {
      b.retract_move();
    }
  }
};

template <board::IndexableView BoardT>
INLINE decltype(auto) make_recursive_move_scope(BoardT &b) {
  return RecursiveMoveScope<BoardT>(b);
}


template <board::IndexableView BoardT>
struct MoveLineScope {
  BoardT &b;
  MoveLine &mline;
  bool is_acting_scope = true;

  INLINE explicit MoveLineScope(BoardT &b, move_t m, MoveLine &mline):
    b(b), mline(mline)
  {
    b.make_move(m);
    mline.premove(m);
  }

  explicit MoveLineScope(const MoveLineScope &other) = delete;
  INLINE explicit MoveLineScope(MoveLineScope &&other):
    b(other.b), mline(other.mline)
  {
    other.is_acting_scope = false;
  }

  INLINE ~MoveLineScope() {
    if(!is_acting_scope)return;
    mline.recall();
    b.retract_move();
  }
};

template <board::IndexableView BoardT>
INLINE decltype(auto) make_mline_scope(BoardT &b, move_t m, MoveLine &mline) {
  return MoveLineScope<BoardT>(b, m, mline);
}


template <board::IndexableView BoardT>
struct RecursiveMoveLineScope {
  BoardT &b;
  MoveLine &mline;
  int counter = 0;
  bool is_acting_scope = true;

  INLINE explicit RecursiveMoveLineScope(BoardT &b, MoveLine &mline):
    b(b), mline(mline)
  {}

  explicit RecursiveMoveLineScope(const RecursiveMoveLineScope &other) = delete;
  INLINE explicit RecursiveMoveLineScope(RecursiveMoveLineScope &&other):
    b(other.b), mline(other.mline), counter(other.counter)
  {
    other.is_acting_scope = false;
  }

  INLINE void scope(move_t m) {
    b.make_move(m);
    mline.premove(m);
    ++counter;
  }

  INLINE void unscope() {
    assert(counter > 0);
    mline.recall();
    b.retract_move();
    --counter;
  }

  INLINE ~RecursiveMoveLineScope() {
    if(!is_acting_scope)return;
    for(int i = 0; i < counter; ++i) {
      mline.recall();
      b.retract_move();
    }
  }
};

template <board::IndexableView BoardT>
INLINE decltype(auto) make_recursive_mline_scope(BoardT &b, MoveLine &mline) {
  return RecursiveMoveLineScope<BoardT>(b, mline);
}
