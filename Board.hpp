#pragma once


#include <vector>

#include <Piece.hpp>
#include <Event.hpp>


// board view of the game
class Board {
private:
  Board &self = *this;
  std::array <Piece *, board::SIZE> board_;
  COLOR activePlayer_;
  std::vector<event_t> history;
public:
  piece_bitboard_t castlings_ = 0x44ULL | 0x44ULL << (board::SIZE - board::LEN);
  pos_t enpassant_ = event::enpassantnotrace;
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
    self[i].unset_pos(i);
    set_pos(i, p);
    return target;
  }

  bool is_castling_move(pos_t i, pos_t j) const {
    if(self[i].value != KING)return false;
    if(self[i].color == WHITE)return Moves<KING, WHITE>::is_castling_move(i, j);
    if(self[i].color == BLACK)return Moves<KING, BLACK>::is_castling_move(i, j);
    return false;
  }

  bool is_enpassant_move(pos_t i, pos_t j) const {
    if(self[i].value != PAWN)return false;
    if(self[i].color == WHITE)return Moves<PAWN, WHITE>::is_enpassant_move(i, j);
    if(self[i].color == BLACK)return Moves<PAWN, BLACK>::is_enpassant_move(i, j);
    return false;
  }

  void move_pos(pos_t i, pos_t j) {
    put_pos(j, self[i]);
    unset_pos(i);
  }

  event_t ev_basic(pos_t i, pos_t j) const {
    assert(!self[i].is_empty());
    pos_t killwhat = event::killnothing;
    pos_t enpassant_trace = event::enpassantnotrace;
    if(self[j].value != EMPTY) {
      killwhat = self[j].piece_index;
    }
    if(is_enpassant_move(i, j)) {
      if(self[i].color==WHITE)enpassant_trace=Moves<PAWN,WHITE>::get_enpassant_trace(i,j);
      if(self[i].color==BLACK)enpassant_trace=Moves<PAWN,BLACK>::get_enpassant_trace(i,j);
    }
    return event::basic(i, j, killwhat, castlings_, enpassant_, enpassant_trace);
  }

  event_t ev_castle(pos_t i, pos_t j) const {
    pos_pair_t rookmove = 0x00;
    if(self[i].color == WHITE)rookmove = Moves<KING,WHITE>::castle_rook_move(i,j);
    if(self[i].color == BLACK)rookmove = Moves<KING,BLACK>::castle_rook_move(i,j);
    pos_t r_i = bitmask::first(rookmove),
          r_j = bitmask::second(rookmove);
    return event::castling(i, j, r_i, r_j, castlings_, enpassant_);
  }

  event_t ev_take_enpassant(pos_t i, pos_t j) const {
    return event::enpassant(i, j, enpassant_layman(), castlings_, enpassant_);
  }

  pos_t enpassant_layman() const {
    if(enpassant_ == event::enpassantnotrace)return 0xFF;
    const pos_t x = board::_x(enpassant_);
    return board::_y(enpassant_ == 3-1) ? board::_pos(A+x, 4) : board::_pos(A+x, 5);
  }

  void reset_enpassants() {
    enpassant_ = 0xFF;
  }

  void update_castlings(pos_t i) {
    const pos_t shift = (self[i].color == WHITE) ? 0 : board::SIZE-board::LEN;
    const piece_bitboard_t castleline = 0xFFULL << shift;
    const piece_bitboard_t castleleft = 0x04ULL << shift;
    const piece_bitboard_t castleright = 0x40ULL << shift;
    if(self[i].value == KING) {
      castlings_ &= ~castleline;
    } else if((castlings_ & castleleft) && self[i].value == ROOK && i == board::_pos(A, 1) + shift) {
      castlings_ &= ~castleleft;
    } else if((castlings_ & castleright) && self[i].value == ROOK && i == board::_pos(H, 1) + shift) {
      castlings_ &= ~castleright;
    }
  }

  void act_event(event_t ev) {
    history.push_back(ev);
    pos_t marker = event::extract_byte(ev);
    assert(event::decompress_castlings(event::compress_castlings(castlings_)) == castlings_);
    switch(marker) {
      case event::BASIC_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          auto killwhat = event::extract_byte(ev);
          auto castlings_ = event::extract_castlings(ev);
          auto enpassant_old = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);
          enpassant_ = enpassant_trace;
          update_castlings(i);
          move_pos(i, j);
        }
      break;
      case event::CASTLING_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t r_i = event::extract_byte(ev);
          pos_t r_j = event::extract_byte(ev);
          auto castlings_ = event::extract_castlings(ev);
          auto enpassant_ = event::extract_byte(ev);
          update_castlings(i);
          move_pos(i, j);
          move_pos(r_i, r_j);
          reset_enpassants();
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t killwhere = event::extract_byte(ev);
          auto castlings_ = event::decompress_castlings(event::extract_byte(ev));
          auto enpassant_ = event::extract_byte(ev);
          put_pos(killwhere, self.get_piece(EMPTY));
          move_pos(i, j);
          reset_enpassants();
        }
      break;
      case event::PROMOTION_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t killwhat = event::extract_byte(ev);
          auto castlings_ = event::extract_castlings(ev);
          auto enpassant_ = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);
          pos_t becomewhat = event::extract_byte(ev);
          put_pos(i, self.get_piece(EMPTY));
          put_pos(j, self.pieces[becomewhat]);
          reset_enpassants();
        }
      break;
    }
    activePlayer_ = enemy_of(activePlayer());
  }

  event_t last_event() const {
    if(history.empty())return 0x00;
    return history.back();
  }

  void unact_event() {
    if(history.empty())return;
    event_t ev = history.back();
    pos_t marker = event::extract_byte(ev);
    switch(marker) {
      case event::BASIC_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t killwhat = event::extract_byte(ev);
          castlings_ = event::extract_castlings(ev);
          move_pos(j, i);
          if(killwhat != event::killnothing) {
            put_pos(j, self.pieces[killwhat]);
          }
          enpassant_ = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);
        }
      break;
      case event::CASTLING_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t r_i = event::extract_byte(ev);
          pos_t r_j = event::extract_byte(ev);
          castlings_ = event::extract_castlings(ev);
          enpassant_ = event::extract_byte(ev);
          move_pos(j, i);
          move_pos(r_j, r_i);
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t killwhere = event::extract_byte(ev);
          castlings_ = event::decompress_castlings(event::extract_byte(ev));
          enpassant_ = event::extract_byte(ev);
          put_pos(killwhere, self.get_piece(PAWN, enemy_of(self[j].color)));
          move_pos(j, i);
          enpassant_ = j;
        }
      break;
      case event::PROMOTION_MARKER:
        {
          pos_t i = event::extract_byte(ev);
          pos_t j = event::extract_byte(ev);
          pos_t killwhat = event::extract_byte(ev);
          castlings_ = event::extract_castlings(ev);
          enpassant_ = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);
          auto becomewhat = event::extract_byte(ev);

          COLOR c = self[j].color;
          if(killwhat != event::killnothing) {
            put_pos(j, self.pieces[killwhat]);
          } else {
            put_pos(j, self.get_piece(EMPTY));
          }
          put_pos(i, self.get_piece(PAWN, c));
        }
      break;
    }
    activePlayer_ = enemy_of(activePlayer());
    history.pop_back();
  }

  void make_move(pos_t i, pos_t j, PIECE as=EMPTY) {
    event_t ev = 0x00;
    if(is_castling_move(i, j)) {
      ev = ev_castle(i, j);
    } else if(self[i].value == PAWN && j == enpassant_) {
      ev = ev_take_enpassant(i, j);
    } else {
      ev = ev_basic(i, j);
    }
    act_event(ev);
  }

  void retract_move() {
    unact_event();
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
    for(int p = 0; p < NO_PIECES; ++p) {
      get_piece((PIECE)p,WHITE).foreach([&](pos_t pos) mutable noexcept -> void {
        moves[pos] |= get_piece((PIECE)p,WHITE).get_moves(pos,friends_white,foes_white, attack_mask_white, castlings_, enpassant_);
      });
      get_piece((PIECE)p,BLACK).foreach([&](pos_t pos) mutable noexcept -> void {
        moves[pos] |= get_piece((PIECE)p,BLACK).get_moves(pos,friends_black,foes_black, attack_mask_black, castlings_, enpassant_);
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
      return self[pos].get_moves(pos,friends_white,foes_white,attack_mask_white,castlings_,enpassant_);
    } else if(self[pos].color==BLACK) {
      const piece_bitboard_t friends_black = get_piece_positions(BLACK);
      const piece_bitboard_t foes_black = get_piece_positions(WHITE, true);
      const piece_bitboard_t attack_mask_black = get_attack_mask(BLACK);
      return self[pos].get_moves(pos,friends_black,foes_black,attack_mask_black,castlings_,enpassant_);
    }
    return 0x00;
  }

  piece_bitboard_t get_attack_mask(COLOR c) const {
    const piece_bitboard_t enemy_foes = get_piece_positions(c, true);
    const piece_bitboard_t enemy_friends = get_piece_positions(enemy_of(c));
    piece_bitboard_t mask = 0x00;
    for(int p = 0; p < NO_PIECES; ++p) {
      mask |= get_piece((PIECE)p, enemy_of(c)).get_attacks(enemy_friends, enemy_foes);
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
