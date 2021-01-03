#pragma once


#include <vector>

#include <Piece.hpp>
#include <Event.hpp>


struct ChangeEvent {
  pos_t from, to;
  event ev_from, ev_to;
  piece_bitboard_t castlings, enpassants;
  pos_t enpassantlayman;
  constexpr ChangeEvent(pos_t from, pos_t to, event ev_from, event ev_to,
                        piece_bitboard_t castlings=0x00, piece_bitboard_t enpassants=0x00,
                        pos_t enpassantlayman=0):
    from(from), to(to), ev_from(ev_from), ev_to(ev_to),
    castlings(castlings), enpassants(enpassants),
    enpassantlayman(enpassantlayman)
  {
    if(ev_from.type == KILL) {
      assert(ev_to.type == DEATH);
    }
  }
};


// board view of the game
class Board {
private:
  Board &self = *this;
  std::array <Piece *, board::SIZE> board_;
  COLOR activePlayer_;
  piece_bitboard_t castlings_ = 0x44ULL | 0x44ULL << (board::SIZE - board::LEN);
public:
  piece_bitboard_t enpassants_ = 0x00ULL;
  pos_t enpassantlayman_ = 0;
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
    Piece(EMPTY, NEUTRAL)
  };
  Board(COLOR activePlayer=WHITE):
    activePlayer_(activePlayer)
  {
    for(pos_t i = 0; i < board::SIZE; ++i) {
      set_pos(i, get_piece(EMPTY, NEUTRAL));
    }
    // make initial position
    for(pos_t i = 0; i < board::LEN; ++i) {
      put_pos(board::_pos(A + i, 2), get_piece(PAWN, WHITE)),
      put_pos(board::_pos(A + i, 7), get_piece(PAWN, BLACK));
    }
    for(const auto &[color, N] : {std::make_pair(WHITE, 1), std::make_pair(BLACK, 8)}) {
      put_pos(board::_pos(A, N), get_piece(ROOK, color)),
//      put_pos(board::_pos(B, N), get_piece(KNIGHT, color)),
//      put_pos(board::_pos(C, N), get_piece(BISHOP, color)),
//      put_pos(board::_pos(D, N), get_piece(QUEEN, color)),
      put_pos(board::_pos(E, N), get_piece(KING, color)),
//      put_pos(board::_pos(F, N), get_piece(BISHOP, color)),
//      put_pos(board::_pos(G, N), get_piece(KNIGHT, color)),
      put_pos(board::_pos(H, N), get_piece(ROOK, color));
    }
  }

  constexpr COLOR activePlayer() const {
    return activePlayer_;
  }

  inline Piece &operator[](pos_t i) {
    return *board_[i];
  }

  inline Piece &at_pos(pos_t i, pos_t j) {
    return *board_[board::_pos(i,j)];
  }

  inline const Piece &operator[](pos_t i) const {
    return *board_[i];
  }

  inline const Piece &at_pos(pos_t i, pos_t j) const {
    return *board_[board::_pos(i,j)];
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
    if(!self[i].is_empty()) {
      p.set_event(KILL);
      self[i].set_event(DEATH);
    }
    self[i].unset_pos(i);
    set_pos(i, p);
    return target;
  }

  bool is_castling_move(pos_t i, pos_t j) {
    if(self[i].value != KING)return false;
    if(self[i].color == WHITE)return Moves<KING, WHITE>::is_castling_move(i, j);
    if(self[i].color == BLACK)return Moves<KING, BLACK>::is_castling_move(i, j);
    return false;
  }

  bool is_enpassant_move(pos_t i, pos_t j) {
    if(self[i].value != PAWN)return false;
    if(self[i].color == WHITE)return Moves<PAWN, WHITE>::is_enpassant_move(i, j);
    if(self[i].color == BLACK)return Moves<PAWN, BLACK>::is_enpassant_move(i, j);
    return false;
  }

  ChangeEvent move(pos_t i, pos_t j) {
    assert(!self[i].is_empty());
    event ev = put_pos(j, self[i]).last_event;
    unset_pos(i);
    return ChangeEvent(i, j, self[j].last_event, ev, castlings_, enpassants_, enpassantlayman_);
  }

  ChangeEvent remove(pos_t i) {
    assert(!self[i].is_empty());
    event ev = put_pos(i, self.get_piece(EMPTY)).last_event;
    return ChangeEvent(i, i, self[i].last_event, ev, castlings_, enpassants_, enpassantlayman_);
  }

  std::vector<ChangeEvent> move_castle(pos_t i, pos_t j) {
    std::vector<ChangeEvent> events;
    pos_pair_t rookmove = 0x00;
    if(self[i].color == WHITE)rookmove = Moves<KING,WHITE>::castle_rook_move(i,j);
    if(self[i].color == BLACK)rookmove = Moves<KING,BLACK>::castle_rook_move(i,j);
    pos_t r_i = bitmask::first(rookmove),
          r_j = bitmask::second(rookmove);
    events.push_back(move(r_i, r_j));
    events.push_back(move(i, j));
    reset_enpassants();
    return events;
  }

  std::vector<ChangeEvent> take_enpassant(pos_t i, pos_t j) {
    std::vector<ChangeEvent> events;
    events.push_back(remove(enpassantlayman_));
    events.push_back(move(i, j));
    reset_enpassants();
    return events;
  }

  ChangeEvent move_enpassant(pos_t i, pos_t j) {
    if(self[i].color==WHITE)enpassants_=Moves<PAWN,WHITE>::get_enpassant_bit(i,j);
    if(self[i].color==BLACK)enpassants_=Moves<PAWN,BLACK>::get_enpassant_bit(i,j);
    enpassantlayman_ = j;
    return move(i, j);
  }

  void reset_enpassants() {
    enpassants_ = 0x00, enpassantlayman_ = 0;
  }

  void update_castling(pos_t i, pos_t j) {
    if(self[i].value == KING) {
      castlings_ &= ~(0xFFULL << ((self[i].color == WHITE) ? 0 : board::SIZE-board::LEN));
    }
  }

  std::vector<ChangeEvent> make_move(pos_t i, pos_t j) {
    std::vector<ChangeEvent> events;
    if(is_castling_move(i, j))events=move_castle(i, j);
    else if(self[i].value == PAWN && (1ULL << j) == enpassants_)events=take_enpassant(i, j);
    else {
      reset_enpassants();
      update_castling(i, j);
      if(is_enpassant_move(i, j))events.push_back(move_enpassant(i, j));
      else events.push_back(move(i, j));
    }
    activePlayer_ = enemy_of(activePlayer());
    return events;
  }

  inline piece_bitboard_t get_piece_positions(COLOR c, bool enemy=false) const {
    piece_bitboard_t mask = UINT64_C(0);
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING})mask|=self.get_piece(p, c).mask;
    if(enemy)mask&=~get_piece(KING, c).mask;
    return mask;
  }

  decltype(auto) get_attacks() const {
    std::array <piece_bitboard_t, board::SIZE> attacks = {UINT64_C(0x00)};
    for(auto&a:attacks)a=UINT64_C(0x00);
    const piece_bitboard_t friends_white = get_piece_positions(WHITE);
    const piece_bitboard_t friends_black = get_piece_positions(BLACK);
    const piece_bitboard_t foes_white = get_piece_positions(BLACK, true);
    const piece_bitboard_t foes_black = get_piece_positions(WHITE, true);
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      get_piece(p,WHITE).foreach([&](pos_t pos) mutable noexcept -> void {
        attacks[pos] |= get_piece(p,WHITE).get_attack(pos,friends_white,foes_white);
      });
      get_piece(p,BLACK).foreach([&](pos_t pos) mutable noexcept -> void {
        attacks[pos] |= get_piece(p,BLACK).get_attack(pos,friends_black,foes_black);
      });
    }
    return attacks;
  }

  inline piece_bitboard_t get_attacks_from(pos_t pos) const {
    if(self[pos].is_empty())return 0x00;
    else if(self[pos].color==WHITE) {
      const piece_bitboard_t friends_white = get_piece_positions(WHITE);
      const piece_bitboard_t foes_white = get_piece_positions(BLACK, true);
      return self[pos].get_attack(pos,friends_white,foes_white);
    } else if(self[pos].color==BLACK) {
      const piece_bitboard_t friends_black = get_piece_positions(BLACK);
      const piece_bitboard_t foes_black = get_piece_positions(WHITE, true);
      return self[pos].get_attack(pos,friends_black,foes_black);
    }
    return 0x00;
  }

  decltype(auto) get_moves() const {
    std::array <piece_bitboard_t, board::SIZE> moves = {UINT64_C(0x00)};
    for(auto&m:moves)m=UINT64_C(0x00);
    const piece_bitboard_t friends_white = get_piece_positions(WHITE);
    const piece_bitboard_t friends_black = get_piece_positions(BLACK);
    const piece_bitboard_t foes_white = get_piece_positions(BLACK, true);
    const piece_bitboard_t foes_black = get_piece_positions(WHITE, true);
    const piece_bitboard_t attack_mask_white = get_attack_mask(WHITE);
    const piece_bitboard_t attack_mask_black = get_attack_mask(BLACK);
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      get_piece(p,WHITE).foreach([&](pos_t pos) mutable noexcept -> void {
        moves[pos] |= get_piece(p,WHITE).get_moves(pos,friends_white,foes_white, attack_mask_white, castlings_, enpassants_);
      });
      get_piece(p,BLACK).foreach([&](pos_t pos) mutable noexcept -> void {
        moves[pos] |= get_piece(p,BLACK).get_moves(pos,friends_black,foes_black, attack_mask_black, castlings_, enpassants_);
      });
    }
    return moves;
  }

  inline piece_bitboard_t get_moves_from(pos_t pos) const {
    if(self[pos].is_empty())return 0x00;
    else if(self[pos].color==WHITE) {
      const piece_bitboard_t friends_white = get_piece_positions(WHITE);
      const piece_bitboard_t foes_white = get_piece_positions(BLACK, true);
      const piece_bitboard_t attack_mask_white = get_attack_mask(WHITE);
      return self[pos].get_moves(pos,friends_white,foes_white,attack_mask_white,castlings_,enpassants_);
    } else if(self[pos].color==BLACK) {
      const piece_bitboard_t friends_black = get_piece_positions(BLACK);
      const piece_bitboard_t foes_black = get_piece_positions(WHITE, true);
      const piece_bitboard_t attack_mask_black = get_attack_mask(BLACK);
      return self[pos].get_moves(pos,friends_black,foes_black,attack_mask_black,castlings_,enpassants_);
    }
    return 0x00;
  }

  piece_bitboard_t get_attack_mask(COLOR c) const {
    const piece_bitboard_t enemy_foes = get_piece_positions(c, true);
    const piece_bitboard_t enemy_friends = get_piece_positions(enemy_of(c));
    piece_bitboard_t mask = 0x00;
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      mask |= get_piece(p, enemy_of(c)).get_attacks(enemy_friends, enemy_foes);
    }
    return mask;
  }

  void print() {
    for(pos_t i = board::LEN; i > 0; --i) {
      for(pos_t j = 0; j < board::LEN; ++j) {
        Piece &p = self[(i-1) * board::LEN + j];
        std::cout << p.str() << " ";
      }
      std::cout << std::endl;
    }
  }
};
