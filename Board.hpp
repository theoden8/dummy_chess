#pragma once


#include <vector>

#include <Piece.hpp>
#include <Event.hpp>
#include <FEN.hpp>


// board view of the game
class Board {
private:
  Board &self = *this;
  std::array <pos_t, board::SIZE> board_;
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
  Board(const fen::FEN f=fen::doublecheck_test_pos):
    activePlayer_(f.active_player),
    castlings_(event::decompress_castlings(f.castling_compressed)),
    enpassant_(f.enpassant)
  {
    for(pos_t i = 0; i < board::SIZE; ++i) {
      set_pos(i, get_piece(EMPTY, NEUTRAL));
    }
    for(pos_t i = 0; i < f.board.length(); ++i) {
      if(f.board[i]==' ')continue;
      COLOR c = islower(f.board[i]) ? BLACK : WHITE;
      PIECE p = EMPTY;
      switch(tolower(f.board[i])) {
        case 'p':p=PAWN;break;
        case 'n':p=KNIGHT;break;
        case 'b':p=BISHOP;break;
        case 'r':p=ROOK;break;
        case 'q':p=QUEEN;break;
        case 'k':p=KING;break;
      }
      pos_t x = board::_x(i),
            y = board::LEN - board::_y(i) - 1;
      put_pos(board::_pos(A+x, 1+y), get_piece(p, c));
    }
    update_state_on_event();
  }

  constexpr COLOR activePlayer() const {
    return activePlayer_;
  }

  inline Piece &operator[](pos_t i) {
    return pieces[board_[i]];
  }

  inline Piece &at_pos(pos_t i, pos_t j) {
    return pieces[board_[board::_pos(i,j)]];
  }

  inline const Piece &operator[](pos_t i) const {
    return pieces[board_[i]];
  }

  inline const Piece &at_pos(pos_t i, pos_t j) const {
    return pieces[board_[board::_pos(i,j)]];
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
    self.board_[i] = p.piece_index;
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
    update_state_on_event(ev);
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
    update_state_on_event(ev, false);
    history.pop_back();
  }

  event_t get_move_event(pos_t i, pos_t j, PIECE as=EMPTY) {
    event_t ev = 0x00;
    if(is_castling_move(i, j)) {
      ev = ev_castle(i, j);
    } else if(self[i].value == PAWN && j == enpassant_) {
      ev = ev_take_enpassant(i, j);
    } else {
      ev = ev_basic(i, j);
    }
    return ev;
  }

  void make_move(pos_t i, pos_t j, PIECE as=EMPTY) {
    act_event(get_move_event(i, j, as));
  }

  void retract_move() {
    unact_event();
  }

  void update_state_on_event(event_t ev=0x00, bool forward=true) {
    update_state_attacks();
    update_state_pins();
    update_state_checkline();
    update_state_moves();
  }

  inline piece_bitboard_t get_piece_positions(COLOR c, bool enemy=false) const {
    piece_bitboard_t mask = UINT64_C(0);
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING})mask|=self.get_piece(p, c).mask;
    if(enemy)mask&=~get_piece(KING, c).mask;
    return mask;
  }

  std::array <piece_bitboard_t, board::SIZE> state_attacks = {UINT64_C(0x00)};
  void update_state_attacks() {
    for(auto&a:state_attacks)a=UINT64_C(0x00);
    const piece_bitboard_t friends_white = get_piece_positions(WHITE);
    const piece_bitboard_t friends_black = get_piece_positions(BLACK);
    const piece_bitboard_t foes_white = get_piece_positions(BLACK, true);
    const piece_bitboard_t foes_black = get_piece_positions(WHITE, true);
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      get_piece(p,WHITE).foreach([&](pos_t pos) mutable noexcept -> void {
        state_attacks[pos] |= get_piece(p,WHITE).get_attack(pos,friends_white,foes_white);
      });
      get_piece(p,BLACK).foreach([&](pos_t pos) mutable noexcept -> void {
        state_attacks[pos] |= get_piece(p,BLACK).get_attack(pos,friends_black,foes_black);
      });
    }
    update_state_attack_counts();
  }

  std::array <pos_t, board::SIZE> state_attacks_count[NO_COLORS] = {{0x00}, {0x00}};
  void update_state_attack_counts() {
    for(COLOR c : {WHITE, BLACK}) {
      for(auto&ac:state_attacks_count[c])ac=0;
      bitmask::foreach(get_piece_positions(c), [&](pos_t i) mutable -> void {
        bitmask::foreach(state_attacks[i], [&](pos_t j) mutable -> void {
          ++state_attacks_count[c][j];
        });
      });
    }
  }

  inline piece_bitboard_t get_attacks_from(pos_t pos) const { return state_attacks[pos]; }

  template <typename F>
  inline void iter_attacking_xrays(pos_t j, F &&func, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)return;
    const piece_bitboard_t friends = get_piece_positions(enemy_of(c)),
                           foes = get_piece_positions(c);
    // apply func only to non-zero rays
    #define APPLY_TO_XRAY(PIECE, COLOR) \
        piece_bitboard_t r = xRayAttacks<PIECE,COLOR>::get_attacking_xray(i, j, friends, foes); \
        if(r != 0x00ULL)func(i, r);
    if(c!=WHITE) {
      get_piece(BISHOP,WHITE).foreach([&](pos_t i)mutable->void{APPLY_TO_XRAY(BISHOP,WHITE);}),
      get_piece(ROOK,WHITE).foreach([&](pos_t i)mutable->void{APPLY_TO_XRAY(ROOK,WHITE);}),
      get_piece(QUEEN,WHITE).foreach([&](pos_t i)mutable->void{APPLY_TO_XRAY(QUEEN,WHITE);});
    } else {
      get_piece(BISHOP,BLACK).foreach([&](pos_t i)mutable->void{APPLY_TO_XRAY(BISHOP,BLACK);}),
      get_piece(ROOK,BLACK).foreach([&](pos_t i)mutable->void{APPLY_TO_XRAY(ROOK,BLACK);}),
      get_piece(QUEEN,BLACK).foreach([&](pos_t i)mutable->void{APPLY_TO_XRAY(QUEEN,BLACK);});
    }
    #undef APPLY_TO_XRAY
  }

  inline piece_bitboard_t get_attacking_xrays(pos_t j, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)return 0x00;
    piece_bitboard_t rtotal = 0x00;
    iter_attacking_xrays(j, [&](pos_t i, piece_bitboard_t r) mutable -> void {
      rtotal |= r;
      rtotal |= 1ULL<<i;
    }, c);
    return rtotal & ~(1ULL << j);
  }

  inline pos_t get_king_pos(COLOR c) const {
    return bitmask::log2_of_exp2(get_piece(KING, c).mask);
  }

  inline piece_bitboard_t get_pins(COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    const pos_t kingpos = get_king_pos(c);
    piece_bitboard_t friends = get_piece_positions(c);
    return friends & get_attacking_xrays(kingpos, c);
  }

  piece_bitboard_t state_pins[NO_COLORS] = {0x00, 0x00};
  std::vector<piece_bitboard_t> state_pins_rays[NO_COLORS];
  std::vector<pos_t> state_pins_attackers[NO_COLORS];
  void update_state_pins() {
    for(COLOR c : {WHITE, BLACK}) {
      state_pins[c] = get_pins(c);

      state_pins_rays[c].clear();
      state_pins_attackers[c].clear();

      const pos_t kingpos = get_king_pos(c);
      pos_t i = 0;
      bitmask::foreach(state_pins[c], [&](pos_t pin) mutable -> void {
        piece_bitboard_t res = 0x00;
        pos_t attacker = 0xFF;
        iter_attacking_xrays(kingpos, [&](pos_t i, piece_bitboard_t r) mutable -> void {
          if(r & (1ULL << pin)) {
            assert(!res);
            res |= r;
            attacker = i;
            res |= 1ULL << attacker;
          }
        }, c);
        state_pins_rays[c].push_back(res);
        state_pins_attackers[c].push_back(attacker);
        ++i;
      });
    }
  }

  piece_bitboard_t get_pin_line_of(pos_t i) const {
    const COLOR c = self[i].color;
    if(c==NEUTRAL)return ~0ULL;
    if(!(state_pins[c] & (1ULL << i)))return ~0ULL;
    for(const auto &r : state_pins_rays[c]) {
      if(r & (1ULL << i))return r;
    }
    return ~0ULL;
  }

  piece_bitboard_t state_checkline = 0x00;
  void update_state_checkline() {
    const COLOR c = activePlayer();
    if(state_attacks_count[enemy_of(c)][get_king_pos(c)] == 0) {
      state_checkline = ~0x00ULL;
      return;
    } else if(state_attacks_count[enemy_of(c)][get_king_pos(c)] >= 2) {
      state_checkline = 0x00;
      return;
    }
    const pos_t kingpos = get_king_pos(c);
    pos_t attacker = 0xFF;
    for(pos_t i=0;i<board::SIZE;++i) {
      if(self[i].color!=enemy_of(c))continue;
      if(state_attacks[i] & (1ULL << kingpos)) {
        attacker = i;
      }
    }
    const PIECE p = self[attacker].value;
    const piece_bitboard_t occupied = get_piece_positions(c) | get_piece_positions(enemy_of(c), true);
    piece_bitboard_t res = 0x00;
    if(p == BISHOP && enemy_of(c) == WHITE)res=Attacks<BISHOP,WHITE>::get_attacking_ray(kingpos,attacker,occupied);
    if(p == ROOK   && enemy_of(c) == WHITE)res=Attacks<ROOK  ,WHITE>::get_attacking_ray(kingpos,attacker,occupied);
    if(p == QUEEN  && enemy_of(c) == WHITE)res=Attacks<QUEEN ,WHITE>::get_attacking_ray(kingpos,attacker,occupied);
    if(p == BISHOP && enemy_of(c) == BLACK)res=Attacks<BISHOP,BLACK>::get_attacking_ray(kingpos,attacker,occupied);
    if(p == ROOK   && enemy_of(c) == BLACK)res=Attacks<ROOK  ,BLACK>::get_attacking_ray(kingpos,attacker,occupied);
    if(p == QUEEN  && enemy_of(c) == BLACK)res=Attacks<QUEEN ,BLACK>::get_attacking_ray(kingpos,attacker,occupied);
    state_checkline = res | (1ULL << attacker);
  }

  std::array <piece_bitboard_t, board::SIZE> state_moves = {UINT64_C(0x00)};
  void update_state_moves() {
    for(auto&m:state_moves)m=UINT64_C(0x00);
    const COLOR c = activePlayer();
    const piece_bitboard_t friends = get_piece_positions(c),
          foes  = get_piece_positions(enemy_of(c), true),
          attack_mask = get_attack_mask(c),
          pins = state_pins[c];
    const bool doublecheck = state_attacks_count[enemy_of(c)][get_king_pos(c)] > 1;
    for(pos_t p = 0; p < NO_PIECES; ++p) {
      if(doublecheck && p != KING)continue;
      get_piece((PIECE)p,c).foreach([&](pos_t pos) mutable noexcept -> void {
        state_moves[pos] = get_piece((PIECE)p,c).get_moves(pos,friends,foes,attack_mask,castlings_,enpassant_);
        if(p != KING)state_moves[pos]&=state_checkline;
        if(pins & (1ULL << pos))state_moves[pos] &= get_pin_line_of(pos);
      });
    }
  }

  inline piece_bitboard_t get_moves_from(pos_t pos) const { return state_moves[pos]; }

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
