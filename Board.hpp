#pragma once

#include "Piece.hpp"

#include <cmath>
#include <cassert>
#include <array>
#include <iostream>

struct ChangeEvent {
  pos_t from, to;
  event ev_from, ev_to;
  constexpr ChangeEvent(pos_t from, pos_t to, event ev_from, event ev_to):
    from(from), to(to), ev_from(ev_from), ev_to(ev_to)
  {
    if(ev_from.type == KILL)
      assert(ev_to.type == DEATH);
  }
};

enum {
  A,B,C,D,E,F,G,H
} ENUM_ROW;

class Board {
public:
  static constexpr const pos_t LENGTH = 8;
  static constexpr const pos_t SIZE = 64;
private:
  Board &self = *this;
  std::array <Piece *, SIZE> board_;
public:
  Board() {
    for(pos_t i = 0; i < SIZE; ++i) {
      board_[i] = &Piece::get(EMPTY);
    }
  }

  constexpr Piece *&operator[](pos_t i) {
    return board_[i];
  }

  static constexpr pos_t _x(pos_t i) {
    return i % LENGTH;
  }

  static constexpr pos_t _y(pos_t i) {
    return i / LENGTH;
  }

  static constexpr pos_t _pos(pos_t i, pos_t j) {
    return i + (j - 1) * LENGTH;
  }

  constexpr void unset_pos(pos_t i) {
    self[i]->unset(i);
    set_pos(i, Piece::get(EMPTY));
  }

  constexpr void set_pos(pos_t i, Piece &p) {
    p.set(i);
    self[i] = &p;
  }

  constexpr Piece &put_pos(pos_t i, Piece &p) {
    Piece &target = *self[i];
    if(!target.is_empty()) {
      p.set_event(KILL);
      target.set_event(DEATH);
    }
    set_pos(i, p);
    return target;
  }

  constexpr ChangeEvent move(pos_t i, pos_t j) {
    assert(!self[i]->is_empty());
    event ev = put_pos(j, *self[i]).last_event;
    self.unset_pos(i);
    return ChangeEvent(i, j, self[j]->last_event, ev);
  }

  void print() {
    for(pos_t i = LENGTH; i > 0; --i) {
      for(pos_t j = 0; j < LENGTH; ++j) {
        Piece &p = *self[(i-1) * LENGTH + j];
        std::cout << p.str() << " ";
      }
      std::cout << std::endl;
    }
  }
};
