#pragma once


#include <Piece.hpp>
#include <Event.hpp>


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


// board view of the game
class Board {
public:
  static constexpr const pos_t LEN = 8;
  static constexpr const pos_t SIZE = 64;
private:
  Board &self = *this;
  std::array <Piece *, SIZE> board_;
public:
  std::array<Piece, 2*6+1>  pieces = {
    Piece(PAWN, WHITE),
    Piece(PAWN, BLACK),
    Piece(KNIGHT, WHITE),
    Piece(KNIGHT, BLACK),
    Piece(BISHOP, WHITE),
    Piece(BISHOP, BLACK),
    Piece(ROOK, WHITE),
    Piece(ROOK, BLACK),
    Piece(QUEEN, WHITE),
    Piece(QUEEN, BLACK),
    Piece(KING, WHITE),
    Piece(KING, BLACK),
    Piece(EMPTY, NEUTRAL, 0xff)
  };
  Board()
  {
    for(pos_t i = 0; i < SIZE; ++i) {
      self.board_[i] = &self.get_piece(EMPTY);
    }
  }

  inline Piece &operator[](pos_t i) {
    return *board_[i];
  }

  inline const Piece &operator[](pos_t i) const {
    return *board_[i];
  }

  const Piece &get_piece(PIECE P = EMPTY, COLOR C = NEUTRAL) const {
    if(P == EMPTY) {
      return pieces[13-1];
    }
    return pieces[(P - PAWN) * 2 + C - WHITE];
  }

  constexpr Piece &get_piece(PIECE P = EMPTY, COLOR C = NEUTRAL) {
    if(P == EMPTY) {
      return pieces[13-1];
    }
    return pieces[(P - PAWN) * 2 + C - WHITE];
  }

  const Piece get_piece(const Piece &p) const {
    return get_piece(p.value, p.color);
  }

  constexpr Piece get_piece(Piece &p) {
    return get_piece(p.value, p.color);
  }

  void unset_pos(pos_t i) {
    self[i].unset_pos(i);
    set_pos(i, self.get_piece(EMPTY));
  }

  void set_pos(pos_t i, Piece &p) {
    p.set_pos(i);
    self.board_[i] = &p;
  }

  Piece &put_pos(pos_t i, Piece &p) {
    Piece &target = self[i];
    if(!target.empty()) {
      p.set_event(KILL);
      target.set_event(DEATH);
    }
    set_pos(i, p);
    return target;
  }

  ChangeEvent move(pos_t i, pos_t j) {
    assert(!self[i].empty());
    event ev = put_pos(j, self[i]).last_event;
    self.unset_pos(i);
    return ChangeEvent(i, j, self[j].last_event, ev);
  }

  piece_bitboard_t get_piece_positions(COLOR c) const {
    piece_bitboard_t mask = UINT64_C(0);
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING})mask |= self.get_piece(p, c).mask;
    return mask;
  }

  void print() {
    for(pos_t i = LEN; i > 0; --i) {
      for(pos_t j = 0; j < LEN; ++j) {
        Piece &p = self[(i-1) * LEN + j];
        std::cout << p.str() << " ";
      }
      std::cout << std::endl;
    }
  }
};


// current game-state
// board: current board
// attacks: currently computed attack moves by each piece
class State {
  Board b;
  COLOR activePlayer_;
public:
  State():
    b(),
    activePlayer_(WHITE)
  {
    for(pos_t i = 0; i < board::LEN; ++i) {
      b.set_pos(board::_pos(A + i, 2), b.get_piece(PAWN, WHITE)),
      b.set_pos(board::_pos(A + i, 7), b.get_piece(PAWN, BLACK));
    }
    // make initial position
    for(const auto &[color, N] : {std::make_pair(WHITE, 1), std::make_pair(BLACK, 8)}) {
      b.set_pos(board::_pos(A, N), b.get_piece(ROOK, color)),
      b.set_pos(board::_pos(B, N), b.get_piece(KNIGHT, color)),
      b.set_pos(board::_pos(C, N), b.get_piece(BISHOP, color)),
      b.set_pos(board::_pos(D, N), b.get_piece(QUEEN, color)),
      b.set_pos(board::_pos(E, N), b.get_piece(KING, color)),
      b.set_pos(board::_pos(F, N), b.get_piece(BISHOP, color)),
      b.set_pos(board::_pos(G, N), b.get_piece(KNIGHT, color)),
      b.set_pos(board::_pos(H, N), b.get_piece(ROOK, color));
    }
    b.set_pos(board::_pos(E, 4), b.get_piece(QUEEN, WHITE));
  }

  decltype(auto) get_attacks() const {
    std::array <piece_bitboard_t, board::SIZE> attacks = {UINT64_C(0x00)};
    for(auto&a:attacks)a=UINT64_C(0x00);
    piece_bitboard_t friends = b.get_piece_positions(activePlayer());
    piece_bitboard_t foes = b.get_piece_positions(enemy_of(activePlayer()));
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      for(COLOR c : {WHITE, BLACK}) {
        b.get_piece(p,c).foreach([&](pos_t pos) mutable noexcept -> void {
          attacks[pos] |= b.get_piece(p,c).get_attack(pos,friends,foes);
        });
      }
    }
    return attacks;
  }

  piece_bitboard_t get_attacks_from(pos_t pos) const {
    return get_attacks()[pos];
  }

  COLOR activePlayer() const {
    return activePlayer_;
  }

  decltype(auto) get_piece(PIECE p, COLOR c) noexcept {
    return b.get_piece(p, c);
  }

  decltype(auto) get_piece(PIECE p, COLOR c) const noexcept {
    return b.get_piece(p, c);
  }

  decltype(auto) at_pos(pos_t ind) noexcept {
    return b[ind];
  }

  decltype(auto) at_pos(pos_t ind) const noexcept {
    return b[ind];
  }

  decltype(auto) at_pos(pos_t i, pos_t j) noexcept {
    return at_pos(board::_pos(i,j));
  }

  decltype(auto) at_pos(pos_t i, pos_t j) const noexcept {
    return at_pos(board::_pos(i,j));
  }

  void print() {
    b.print();
  }
};
