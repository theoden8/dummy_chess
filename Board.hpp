#pragma once


#include <vector>

#include <Piece.hpp>
#include <Event.hpp>
#include <FEN.hpp>


// board view of the game
class Board {
private:
  static bool m42_initialized;
  std::array <pos_t, board::SIZE> board_;
  COLOR activePlayer_;
  std::vector<event_t> history;
public:
  Board &self = *this;
  piece_bitboard_t castlings_ = 0x44ULL | 0x44ULL << (board::SIZE - board::LEN);
  pos_t enpassant_ = event::enpassantnotrace;
  pos_t halfmoves_ = 0;
  std::array<Piece, 2*6+1>  pieces = {
    Piece(PAWN, WHITE), Piece(PAWN, BLACK),
    Piece(KNIGHT, WHITE), Piece(KNIGHT, BLACK),
    Piece(BISHOP, WHITE), Piece(BISHOP, BLACK),
    Piece(ROOK, WHITE), Piece(ROOK, BLACK),
    Piece(QUEEN, WHITE), Piece(QUEEN, BLACK),
    Piece(KING, WHITE), Piece(KING, BLACK),
    Piece(EMPTY, NEUTRAL)
  };
  Board(const fen::FEN f):
    activePlayer_(f.active_player),
    castlings_(event::decompress_castlings(f.castling_compressed)),
    enpassant_(f.enpassant),
    halfmoves_(f.halfmove_clock)
  {
    if(!m42_initialized) {
      M42::init();
      m42_initialized = true;
    }
    for(pos_t i = 0; i < board::SIZE; ++i) {
      set_pos(i, get_piece(EMPTY, NEUTRAL));
    }
    for(pos_t i = 0; i < f.board.length(); ++i) {
      if(f.board[i]==' ')continue;
      const COLOR c = islower(f.board[i]) ? BLACK : WHITE;
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
    init_update_state();
  }

  inline constexpr COLOR activePlayer() const {
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

  inline const Piece &get_piece(PIECE p=EMPTY, COLOR c=NEUTRAL) const {
    return pieces[Piece::get_piece_index(p, c)];
  }

  inline constexpr Piece &get_piece(PIECE p=EMPTY, COLOR c=NEUTRAL) {
    return pieces[Piece::get_piece_index(p, c)];
  }

  inline const Piece get_piece(const Piece &p) const {
    return get_piece(p.value, p.color);
  }

  inline constexpr Piece get_piece(Piece &p) {
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

  void put_pos(pos_t i, Piece &p) {
    self[i].unset_pos(i);
    set_pos(i, p);
  }

  inline bool is_castling_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    if(self[i].value != KING)return false;
    if(self[i].color == WHITE)return Moves<KINGM>::is_castling_move<WHITE>(i, j);
    if(self[i].color == BLACK)return Moves<KINGM>::is_castling_move<BLACK>(i, j);
    return false;
  }

  inline bool is_enpassant_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    if(self[i].value != PAWN)return false;
    if(self[i].color == WHITE)return Moves<WPAWNM>::is_enpassant_move(i, j);
    if(self[i].color == BLACK)return Moves<BPAWNM>::is_enpassant_move(i, j);
    return false;
  }

  inline bool is_promotion_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    if(self[i].value != PAWN)return false;
    if(self[i].color == WHITE)return Moves<WPAWNM>::is_promotion_move(i, j);
    if(self[i].color == BLACK)return Moves<BPAWNM>::is_promotion_move(i, j);
    return false;
  }

  void move_pos(pos_t i, pos_t j) {
    assert(j <= board::MOVEMASK);
    put_pos(j, self[i]);
    unset_pos(i);
  }

  event_t ev_basic(pos_t i, pos_t j) const {
    const pos_t promote_flag = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    assert(!self[i].is_empty());
    pos_t killwhat = event::killnothing;
    pos_t enpassant_trace = event::enpassantnotrace;
    if(self[j].value != EMPTY) {
      killwhat = self[j].piece_index;
    }
    if(is_enpassant_move(i, j)) {
      if(self[i].color==WHITE)enpassant_trace=Moves<WPAWNM>::get_enpassant_trace(i,j);
      if(self[i].color==BLACK)enpassant_trace=Moves<BPAWNM>::get_enpassant_trace(i,j);
    }
    return event::basic(i, j | promote_flag, killwhat, castlings_, halfmoves_, enpassant_, enpassant_trace);
  }

  event_t ev_castle(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    pos_pair_t rookmove = 0x00;
    if(self[i].color == WHITE)rookmove = Moves<KINGM>::castle_rook_move<WHITE>(i,j);
    if(self[i].color == BLACK)rookmove = Moves<KINGM>::castle_rook_move<BLACK>(i,j);
    const pos_t r_i = bitmask::first(rookmove),
                r_j = bitmask::second(rookmove);
    return event::castling(i, j, r_i, r_j, castlings_, halfmoves_, enpassant_);
  }

  event_t ev_take_enpassant(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    return event::enpassant(i, j, enpassant_pawn(), castlings_, halfmoves_, enpassant_);
  }

  pos_t enpassant_pawn() const {
    if(enpassant_ == event::enpassantnotrace)return 0xFF;
    const pos_t x = board::_x(enpassant_);
    return board::_y(enpassant_) == 3-1 ? board::_pos(A+x, 4) : board::_pos(A+x, 5);
  }

  inline PIECE get_promotion_as(pos_t j) const {
    switch(j & ~board::MOVEMASK) {
      case board::PROMOTE_KNIGHT:return KNIGHT;
      case board::PROMOTE_BISHOP:return BISHOP;
      case board::PROMOTE_ROOK:return ROOK;
      case board::PROMOTE_QUEEN:return QUEEN;
    }
    return PAWN;
  }

  event_t ev_promotion(pos_t i, pos_t j) const {
    return event::promotion_from_basic(ev_basic(i, j));
  }

  void reset_enpassants() {
    enpassant_ = 0xFF;
  }

  void update_castlings(pos_t i, pos_t j) {
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

    const pos_t antishift = (self[i].color == BLACK) ? 0 : board::SIZE-board::LEN;
    const piece_bitboard_t anticastleleft = 0x04ULL << antishift;
    const piece_bitboard_t anticastleright = 0x40ULL << antishift;
    if((castlings_ & anticastleleft) && self[j].value == ROOK && j == board::_pos(A, 1) + antishift) {
      castlings_ &= ~anticastleleft;
    } else if((castlings_ & anticastleright) && self[j].value == ROOK && j == board::_pos(H, 1) + antishift) {
      castlings_ &= ~anticastleright;
    }
  }

  void act_event(event_t ev) {
    history.push_back(ev);
    const pos_t marker = event::extract_byte(ev);
    assert(event::decompress_castlings(event::compress_castlings(castlings_)) == castlings_);
    switch(marker) {
      case event::BASIC_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          const pos_t killwhat = event::extract_byte(ev);
          auto castlings_ = event::extract_castlings(ev);
          auto halfmoves = event::extract_byte(ev);
          auto enpassant_old = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);
          enpassant_ = enpassant_trace;
          if(killwhat == event::killnothing)++halfmoves_;
          else halfmoves_ = 0;
          update_castlings(i, j);
          move_pos(i, j);
          update_state_pos(i);
          update_state_pos(j);
        }
      break;
      case event::CASTLING_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          const pos_t r_i = event::extract_byte(ev);
          const pos_t r_j = event::extract_byte(ev);
          auto castlings_ = event::extract_castlings(ev);
          auto halfmoves = event::extract_byte(ev);
          auto enpassant_ = event::extract_byte(ev);
          ++halfmoves_;
          update_castlings(i, j);
          move_pos(i, j);
          update_state_pos(i);
          update_state_pos(j);
          move_pos(r_i, r_j);
          update_state_pos(r_i);
          update_state_pos(r_j);
          reset_enpassants();
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          const pos_t killwhere = event::extract_byte(ev);
          auto castlings_ = event::extract_castlings(ev);
          auto halfmoves = event::extract_byte(ev);
          auto enpassant_ = event::extract_byte(ev);
          put_pos(killwhere, self.get_piece(EMPTY));
          update_state_attacks_pos(killwhere);
          update_state_pos(killwhere);
          move_pos(i, j);
          update_state_pos(i);
          update_state_pos(j);
          reset_enpassants();
          halfmoves_ = 0;
        }
      break;
      case event::PROMOTION_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t to_byte = event::extract_byte(ev);
            const pos_t j = to_byte & board::MOVEMASK;
            const PIECE becomewhat = get_promotion_as(to_byte);
          const pos_t killwhat = event::extract_byte(ev);
          auto castlings_ = event::extract_castlings(ev);
          auto halfmoves = event::extract_byte(ev);
          auto enpassant_ = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);
          update_castlings(i, j);
          put_pos(i, self.get_piece(EMPTY));
          put_pos(j, self.pieces[Piece::get_piece_index(becomewhat, activePlayer())]);
          update_state_pos(i);
          update_state_pos(j);
          reset_enpassants();
          halfmoves_ = 0;
        }
      break;
    }
    activePlayer_ = enemy_of(activePlayer());
    update_state_on_event(ev, true);
  }

  inline event_t last_event() const {
    if(history.empty())return 0x00;
    return history.back();
  }

  void unact_event() {
    if(history.empty())return;
    event_t ev = history.back();
    const pos_t marker = event::extract_byte(ev);
    switch(marker) {
      case event::BASIC_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          const pos_t killwhat = event::extract_byte(ev);
          castlings_ = event::extract_castlings(ev);
          halfmoves_ = event::extract_byte(ev);
          move_pos(j, i);
          update_state_pos(i);
          if(killwhat != event::killnothing) {
            put_pos(j, self.pieces[killwhat]);
          }
          update_state_pos(j);
          enpassant_ = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);
        }
      break;
      case event::CASTLING_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          const pos_t r_i = event::extract_byte(ev);
          const pos_t r_j = event::extract_byte(ev);
          castlings_ = event::extract_castlings(ev);
          halfmoves_ = event::extract_byte(ev);
          enpassant_ = event::extract_byte(ev);
          move_pos(j, i);
          update_state_pos(j);
          update_state_pos(i);
          move_pos(r_j, r_i);
          update_state_pos(r_j);
          update_state_pos(r_i);
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev);
          const pos_t killwhere = event::extract_byte(ev);
          castlings_ = event::extract_castlings(ev);
          halfmoves_ = event::extract_byte(ev);
          enpassant_ = event::extract_byte(ev);
          put_pos(killwhere, self.get_piece(PAWN, enemy_of(self[j].color)));
          update_state_pos(killwhere);
          move_pos(j, i);
          update_state_pos(j);
          update_state_pos(i);
          enpassant_ = j;
        }
      break;
      case event::PROMOTION_MARKER:
        {
          const pos_t i = event::extract_byte(ev);
          const pos_t j = event::extract_byte(ev) & board::MOVEMASK;
          const pos_t killwhat = event::extract_byte(ev);
          castlings_ = event::extract_castlings(ev);
          halfmoves_ = event::extract_byte(ev);
          enpassant_ = event::extract_byte(ev);
          auto enpassant_trace = event::extract_byte(ev);

          const COLOR c = self[j].color;
          if(killwhat != event::killnothing) {
            put_pos(j, self.pieces[killwhat]);
          } else {
            put_pos(j, self.get_piece(EMPTY));
          }
          update_state_pos(j);
          put_pos(i, self.get_piece(PAWN, c));
          update_state_pos(i);
        }
      break;
    }
    activePlayer_ = enemy_of(activePlayer());
    history.pop_back();
    update_state_on_event(ev, false);
  }

  event_t get_move_event(pos_t i, pos_t j) const {
    event_t ev = 0x00;
    const pos_t promote_as = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    if(is_castling_move(i, j)) {
      ev = ev_castle(i, j);
    } else if(self[i].value == PAWN && j == enpassant_) {
      ev = ev_take_enpassant(i, j);
    } else if(is_promotion_move(i, j)) {
      ev = ev_promotion(i, j | promote_as);
    } else {
      ev = ev_basic(i, j & board::MOVEMASK);
    }
    return ev;
  }

  void make_move(pos_t i, pos_t j) {
    act_event(get_move_event(i, j));
  }

  void retract_move() {
    unact_event();
  }

  void init_update_state() {
    init_state_attacks();
    init_state_checkline();
    init_state_moves();
  }

  void update_state_pos(pos_t pos) {
    update_state_attacks_pos(pos);
  }

  void update_state_on_event(event_t ev=0x00, bool forward=true) {
    if(forward) {
      update_state_checkline();
      init_state_moves();
    } else {
      update_state_checkline();
      init_state_moves();
    }
  }

  inline piece_bitboard_t get_piece_positions(COLOR c, bool remove_king=false) const {
    if(c==NEUTRAL)return get_piece(EMPTY).mask;
    if(c==BOTH)return ~get_piece_positions(NEUTRAL);
    piece_bitboard_t mask = 0ULL;
    for(PIECE p : {PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING}) {
      mask|=self.get_piece(p, c).mask;
    }
    if(remove_king) {
      mask&=~get_piece(KING, c).mask;
    }
    return mask;
  }

  std::array <piece_bitboard_t, board::SIZE> state_attacks = {UINT64_C(0x00)};
  void init_state_attacks() {
    for(auto&a:state_attacks)a=UINT64_C(0x00);
    for(COLOR c : {WHITE, BLACK}) {
      const piece_bitboard_t occupied = get_piece_positions(BOTH) & ~get_piece(KING, enemy_of(c)).mask;
      for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
        get_piece(p,c).foreach([&](pos_t pos) mutable noexcept -> void {
          state_attacks[pos] |= get_piece(p,c).get_attack(pos,occupied);
        });
      }
    }
  }

  inline piece_bitboard_t get_attacks_from(pos_t pos) const { return state_attacks[pos]; }

  inline piece_bitboard_t get_sliding_attacks_to(pos_t j, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return get_attacks_to(j,WHITE)|get_attacks_to(j,BLACK);
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    return (Attacks<BISHOPM>::get_attacks(j,occupied) & (get_piece(BISHOP,c).mask|get_piece(QUEEN,c).mask))
        | (Attacks<ROOKM>::get_attacks(j,occupied) & (get_piece(ROOK,c).mask|get_piece(QUEEN,c).mask));
  }

  inline piece_bitboard_t get_attacks_to(pos_t j, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return get_attacks_to(j,WHITE)|get_attacks_to(j,BLACK);
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    return get_sliding_attacks_to(j, c)
        | (Attacks<KNIGHTM>::get_attacks(j) & get_piece(KNIGHT,c).mask)
        | (Attacks<KINGM>::get_attacks(j,occupied) & get_piece(KING,c).mask)
        | (get_piece(PAWN,enemy_of(c)).get_attack(j,occupied) & get_piece(PAWN,c).mask);
  }

  inline piece_bitboard_t get_attack_counts_to(pos_t j, COLOR c=NEUTRAL) const {
    return bitmask::count_bits(get_attacks_to(j,c));
  }

  void update_state_attacks_pos(pos_t pos) {
    const piece_bitboard_t affected = (1ULL << pos) | get_sliding_attacks_to(pos, BOTH);
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    bitmask::foreach(affected, [&](pos_t i) mutable -> void {
      state_attacks[i] = self[i].get_attack(i, occupied & ~get_piece(KING,enemy_of(self[i].color)).mask);
    });
  }

  inline piece_bitboard_t get_attack_mask(COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return get_attack_mask(WHITE)|get_attack_mask(BLACK);
    const piece_bitboard_t occupied = get_piece_positions(BOTH) & ~get_piece(KING, enemy_of(c)).mask;
    piece_bitboard_t mask = 0x00;
    for(int p = 0; p < NO_PIECES; ++p) {
      mask |= get_piece((PIECE)p, c).get_attacks(occupied);
    }
    return mask;
  }

  inline pos_t get_king_pos(COLOR c) const {
    return bitmask::log2_of_exp2(get_piece(KING, c).mask);
  }

  piece_bitboard_t state_checkline[NO_COLORS] = {0x00,0x00};
  void init_state_checkline(COLOR c=NEUTRAL) {
    if(c==NEUTRAL)c=activePlayer();
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    const pos_t kingpos = get_king_pos(c);
    const piece_bitboard_t attackers = get_attacks_to(kingpos, enemy_of(c));
    if(!attackers) {
      state_checkline[c] = ~0x00ULL;
      return;
    } else if(!bitmask::is_exp2(attackers)) {
      state_checkline[c] = 0x00;
      return;
    }
    const pos_t attacker = bitmask::log2_of_exp2(attackers);
    state_checkline[c] = self[attacker].get_attacking_ray(kingpos,attacker,occupied&~get_piece(KING,c).mask);
    state_checkline[c] |= (1ULL << attacker);
  }

  piece_bitboard_t state_update_checkline_kingattacks[NO_COLORS] = {~0ULL,~0ULL};
  inline void update_state_checkline() {
    const COLOR c = activePlayer();
    const piece_bitboard_t attackers = get_attacks_to(get_king_pos(c), enemy_of(c));
    if(attackers == state_update_checkline_kingattacks[c])return;
    state_update_checkline_kingattacks[c]=attackers;
    init_state_checkline(c);
  }

  // fifty-halfmoves-draw
  inline bool is_draw_halfmoves() const {
    return halfmoves_ == 50;
  }

  inline bool can_move(COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    bool canmove = false;
    for(PIECE p : {PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING}) {
      get_piece(p,c).foreach([&](pos_t i) mutable -> void {
          if(get_moves_from(i,c)) {
            canmove = true;
          }
      });
      if(canmove)break;
    }
    return canmove;
  }

  inline bool is_draw_stalemate() const {
    const COLOR c = activePlayer();
    return !get_attacks_to(get_king_pos(c), enemy_of(c)) && !can_move(c);
  }

  inline bool is_draw_material() const {
    return (bitmask::count_bits(get_piece_positions(BOTH)) == 2) // two kings
        || (bitmask::count_bits(get_piece_positions(BOTH)) == 3 // material draw, kings and a bishop/knight
            && get_piece(KNIGHT,WHITE).size() + get_piece(BISHOP,WHITE).size()
               + get_piece(KNIGHT,BLACK).size() + get_piece(BISHOP,BLACK).size() == 1)
        || (bitmask::count_bits(get_piece_positions(BOTH)) == 4 // material draw, kings and a bishop/knight each
            && get_piece(KNIGHT,WHITE).size() + get_piece(BISHOP,WHITE).size() == 1
            && get_piece(KNIGHT,BLACK).size() + get_piece(BISHOP,BLACK).size() == 1);
  }

  inline bool is_draw() const {
    return is_draw_halfmoves() || is_draw_material() || is_draw_stalemate();
  }

  std::array <piece_bitboard_t, board::SIZE> state_moves = {0x00};
  void init_state_moves() {
    for(auto&m:state_moves)m=0x00;
    if(is_draw_halfmoves()||is_draw_material())return;
    const COLOR c = activePlayer();
    // maybe do this as a loop for incremental updates
    {
      const piece_bitboard_t friends = get_piece_positions(c),
                             foes  = get_piece_positions(enemy_of(c)),
                             attack_mask = get_attack_mask(enemy_of(c)),
                             pins = get_pins(c);
      const bool doublecheck = (state_checkline[c] == 0x00);
      for(pos_t p = 0; p < NO_PIECES; ++p) {
        if(doublecheck && p!=KING)continue;
        get_piece((PIECE)p,c).foreach([&](pos_t pos) mutable noexcept -> void {
          if(p==PAWN) {
            state_moves[pos] = get_attacks_from(pos) & (foes|(1ULL << enpassant_));
            state_moves[pos] |= get_push_moves(c, pos, friends|foes);
            state_moves[pos] &= state_checkline[c];
          } else if(p==KING) {
            state_moves[pos] = get_attacks_from(pos) & ~friends & ~attack_mask;
            state_moves[pos] |= Moves<KINGM>::get_castling_moves(pos, friends|foes, attack_mask, castlings_, c);
          } else {
            state_moves[pos] = get_attacks_from(pos) & ~friends;
            state_moves[pos] &= state_checkline[c];
          }
        });
      }
      init_state_pins();
    }
  }

  inline piece_bitboard_t get_pins(COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return get_pins(WHITE)|get_pins(BLACK);
    return state_pins[c];
  }

  template <typename F>
  inline void iter_attacking_xrays(pos_t j, F &&func, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)return;
    const piece_bitboard_t dstbit = 1ULL << j;
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t friends = get_piece_positions(c),
                           foes = get_piece_positions(ec);
    const piece_bitboard_t rook_xray = get_piece(ROOK,c).get_xray_attack(j,foes,friends)
                                       & (get_piece(ROOK,ec).mask | get_piece(QUEEN,ec).mask);
    if(rook_xray) {
      bitmask::foreach(rook_xray, [&](pos_t i) mutable -> void {
        if(self[i].value == ROOK || self[i].value == QUEEN) {
          const piece_bitboard_t srcbit = 1ULL << i;
          const piece_bitboard_t r = get_piece(ROOK,c).get_attacking_xray(i,j,foes|dstbit,friends) & ~srcbit;
          func(i, r | dstbit);
        }
      });
    }
    const piece_bitboard_t bishop_xray = get_piece(BISHOP,c).get_xray_attack(j,foes,friends)
                                         & (get_piece(BISHOP,ec).mask | get_piece(QUEEN,ec).mask);
    if(bishop_xray) {
      bitmask::foreach(bishop_xray, [&](pos_t i) mutable -> void {
        if(self[i].value == BISHOP || self[i].value == QUEEN) {
          const piece_bitboard_t srcbit = 1ULL << i;
          const piece_bitboard_t r = get_piece(BISHOP,c).get_attacking_xray(i,j,foes|dstbit,friends) & ~srcbit;
          func(i, r | dstbit);
        }
      });
    }
  }

  piece_bitboard_t state_pins[NO_COLORS] = {0x00, 0x00};
  std::array<piece_bitboard_t, board::SIZE> state_pins_rays;
  void init_state_pins() {
    for(auto &spr:state_pins_rays)spr=~0x00ULL;
    const COLOR c = activePlayer();
    // maybe do this as a loop for incremental updates
    {
      state_pins[c] = 0x00;
      const piece_bitboard_t friends = get_piece_positions(c) & ~get_piece(KING,c).mask;
      iter_attacking_xrays(get_king_pos(c), [&](pos_t i, piece_bitboard_t r) mutable -> void {
        const pos_t attacker = i;
        r |= 1ULL << attacker;
        r &= ~get_piece(KING,c).mask;
        const piece_bitboard_t pin = friends & r;
        if(pin) {
          assert(bitmask::is_exp2(pin));
          pos_t pin_pos = bitmask::log2_of_exp2(pin);
          state_pins[c] |= pin;
          state_pins_rays[pin_pos] = r;
          // update moves
          state_moves[pin_pos] &= r;
        }
      }, c);
    }
  }

  inline piece_bitboard_t get_moves_from(pos_t pos, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return state_moves[pos];
    if(self[pos].color!=c)return 0x00;
    return state_moves[pos];
  }

  void print() const {
    for(pos_t i = board::LEN; i > 0; --i) {
      for(pos_t j = 0; j < board::LEN; ++j) {
        const Piece &p = self[(i-1) * board::LEN + j];
        std::cout << p.str() << " ";
      }
      std::cout << std::endl;
    }
  }

  fen::FEN export_as_fen() const {
    fen::FEN f = {
      .board = std::string(),
      .active_player = activePlayer(),
      .castling_compressed = event::compress_castlings(castlings_),
      .enpassant = enpassant_,
      .halfmove_clock = halfmoves_,
      .fullmove = uint16_t(history.size() / 2),
    };
    for(pos_t y_ = 0; y_ < board::LEN; ++y_) {
      const pos_t y = board::LEN - y_ - 1;
      pos_t emptycount = 0;
      for(pos_t x = 0; x < board::LEN; ++x) {
        const pos_t ind = board::_pos(A+x, 1+y);
        if(self[ind].value == EMPTY) {
          ++emptycount;
          continue;
        }
        if(emptycount > 0) {
          f.board += std::to_string(emptycount);
          emptycount = 0;
        }
        f.board += self[ind].str();
      }
      if(emptycount > 0) {
        f.board += std::to_string(emptycount);
        emptycount = 0;
      }
      if(y != 0)f.board+="/";
    }
    return f;
  }
};

bool Board::m42_initialized = false;

fen::FEN fen::export_from_board(const Board &board) {
  return board.export_as_fen();
}
