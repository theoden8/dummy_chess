#pragma once


#include <vector>
#include <list>

#include <Piece.hpp>
#include <Event.hpp>
#include <FEN.hpp>
#include <Zobrist.hpp>

#include <MoveGuard.hpp>
#include <MoveLine.hpp>


// board view of the game
class Board {
protected:
  static bool initialized;
  std::array <move_index_t, board::SIZE> board_;
  COLOR activePlayer_;
  std::vector<event_t> history;

  struct board_info {
    std::array<piece_bitboard_t, 12> piecemasks;
    COLOR active_player;
    pos_t enpassant_castlings;

    INLINE bool operator==(board_info other) const noexcept {
      return active_player == other.active_player && enpassant_castlings == other.enpassant_castlings && piecemasks == other.piecemasks;
    }

    INLINE void unset() {
      active_player = NEUTRAL;
    }
  };
public:
  Board &self = *this;
  std::vector<std::pair<move_index_t, pos_t>> enpassants;
  std::array<move_index_t, 4> castlings = {board::nocastlings, board::nocastlings, board::nocastlings, board::nocastlings};
  std::vector<move_index_t> halfmoves;
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
    halfmoves({f.halfmove_clock})
  {
    if(!initialized) {
      M42::init();
      zobrist::init();
      initialized = true;
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
      const pos_t x = board::_x(i),
                  y = board::LEN - board::_y(i) - 1;
      put_pos(board::_pos(A+x, 1+y), get_piece(p, c));
    }
    if(!(f.castling_compressed & (1 << 0)))unset_castling(BLACK,QUEEN_SIDE);
    if(!(f.castling_compressed & (1 << 1)))unset_castling(BLACK,KING_SIDE);
    if(!(f.castling_compressed & (1 << 2)))unset_castling(WHITE,QUEEN_SIDE);
    if(!(f.castling_compressed & (1 << 3)))unset_castling(WHITE,KING_SIDE);
    init_update_state();
    if(f.enpassant != event::enpassantnotrace) {
      enpassants.emplace_back(get_current_ply(), f.enpassant);
    }
  }

  INLINE constexpr COLOR activePlayer() const {
    return activePlayer_;
  }

  INLINE Piece &operator[](pos_t ind) {
    assert(!(ind & ~board::MOVEMASK));
    return pieces[board_[ind]];
  }

  INLINE Piece &at_pos(pos_t i, pos_t j) {
    return pieces[board_[board::_pos(i,j)]];
  }

  INLINE const Piece &operator[](pos_t ind) const {
    assert(!(ind & ~board::MOVEMASK));
    return pieces[board_[ind]];
  }

  INLINE const Piece &at_pos(pos_t i, pos_t j) const {
    return pieces[board_[board::_pos(i,j)]];
  }

  INLINE const Piece &get_piece(PIECE p=EMPTY, COLOR c=NEUTRAL) const {
    return pieces[Piece::get_piece_index(p, c)];
  }

  INLINE constexpr Piece &get_piece(PIECE p=EMPTY, COLOR c=NEUTRAL) {
    return pieces[Piece::get_piece_index(p, c)];
  }

  INLINE const Piece get_piece(const Piece &p) const {
    return get_piece(p.value, p.color);
  }

  INLINE constexpr Piece get_piece(Piece &p) {
    return get_piece(p.value, p.color);
  }

  INLINE board_info get_board_info() const {
    pos_t enpassant_castlings = (((enpassant_trace() == event::enpassantnotrace) ? 0x0f : board::_x(enpassant_trace())) << 4);
    enpassant_castlings |= get_castlings_compressed();
    return (board_info){
      .piecemasks = {
        get_piece(PAWN,WHITE).mask, get_piece(KNIGHT,WHITE).mask, get_piece(BISHOP,WHITE).mask,
        get_piece(ROOK,WHITE).mask, get_piece(QUEEN,WHITE).mask, get_piece(KING,WHITE).mask,
        get_piece(PAWN,BLACK).mask, get_piece(KNIGHT,BLACK).mask, get_piece(BISHOP,BLACK).mask,
        get_piece(ROOK,BLACK).mask, get_piece(QUEEN,BLACK).mask, get_piece(KING,BLACK).mask
      },
      .active_player = activePlayer(),
      .enpassant_castlings = enpassant_castlings
    };
  }

  INLINE void unset_pos(pos_t i) {
    self[i].unset_pos(i);
    set_pos(i, self.get_piece(EMPTY));
  }

  INLINE void set_pos(pos_t i, Piece &p) {
    p.set_pos(i);
    self.board_[i] = p.piece_index;
  }

  INLINE void put_pos(pos_t i, Piece &p) {
    self[i].unset_pos(i);
    set_pos(i, p);
  }

  INLINE bool is_castling_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    if(self[i].value != KING)return false;
    if(self[i].color == WHITE)return Moves<KINGM>::is_castling_move<WHITE>(i, j);
    if(self[i].color == BLACK)return Moves<KINGM>::is_castling_move<BLACK>(i, j);
    return false;
  }

  INLINE bool is_doublepush_move(pos_t i, pos_t j) const {
    j &= board::MOVEMASK;
    if(self[i].value != PAWN)return false;
    if(self[i].color == WHITE)return Moves<WPAWNM>::is_double_push(i, j);
    if(self[i].color == BLACK)return Moves<BPAWNM>::is_double_push(i, j);
    return false;
  }

  INLINE bool is_enpassant_take_move(pos_t i, pos_t j) const {
    j &= board::MOVEMASK;
    return self[i].value == PAWN && j == enpassant_trace() && j != event::enpassantnotrace;
  }

  INLINE bool is_promotion_move(pos_t i, pos_t j) const {
    j &= board::MOVEMASK;
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
    if(is_doublepush_move(i, j)) {
      if(self[i].color==WHITE)enpassant_trace=Moves<WPAWNM>::get_enpassant_trace(i,j);
      if(self[i].color==BLACK)enpassant_trace=Moves<BPAWNM>::get_enpassant_trace(i,j);
    }
    return event::basic(bitmask::_pos_pair(i, j | promote_flag), killwhat, enpassant_trace);
  }

  event_t ev_castle(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    pos_pair_t rookmove = 0x00;
    if(self[i].color == WHITE)rookmove = Moves<KINGM>::castle_rook_move<WHITE>(i,j);
    if(self[i].color == BLACK)rookmove = Moves<KINGM>::castle_rook_move<BLACK>(i,j);
    const pos_t r_i = bitmask::first(rookmove),
                r_j = bitmask::second(rookmove);
    return event::castling(bitmask::_pos_pair(i, j), bitmask::_pos_pair(r_i, r_j));
  }

  event_t ev_take_enpassant(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    return event::enpassant(bitmask::_pos_pair(i, j), enpassant_pawn());
  }

  INLINE pos_t enpassant_trace() const {
    if(enpassants.empty())return event::enpassantnotrace;
    const auto [ply, e] = enpassants.back();
    if(ply == get_current_ply()) {
      return e;
    }
    return event::enpassantnotrace;
  }

  INLINE pos_t enpassant_pawn() const {
    if(enpassant_trace() == event::enpassantnotrace)return 0xFF;
    const pos_t x = board::_x(enpassant_trace());
    return board::_y(enpassant_trace()) == 3-1 ? board::_pos(A+x, 4) : board::_pos(A+x, 5);
  }

  void set_enpassant(pos_t e) {
    enpassants.emplace_back(get_current_ply(), e);
  }

  INLINE void restore_enpassants() {
    while(!enpassants.empty() && enpassants.back().first > get_current_ply()) {
      enpassants.pop_back();
    }
  }

  INLINE event_t ev_promotion(pos_t i, pos_t j) const {
    return event::promotion_from_basic(ev_basic(i, j));
  }

  INLINE pos_t get_halfmoves() const {
    if(halfmoves.size() == 1) {
      return halfmoves.front();
    }
    return get_current_ply() - halfmoves.back();
  }

  INLINE void update_halfmoves() {
    halfmoves.emplace_back(get_current_ply());
  }

  INLINE void restore_halfmoves() {
    if(halfmoves.size() == 1)return;
    while(halfmoves.back() > get_current_ply()) {
      halfmoves.pop_back();
    }
  }

  INLINE bool is_castling(COLOR c, CASTLING_SIDE side) const {
    return castlings[board::_castling_index(c, side)] == board::nocastlings;
  }

  INLINE void set_castling(COLOR c, CASTLING_SIDE side) {
    castlings[board::_castling_index(c, side)] = board::nocastlings;
  }

  INLINE void unset_castling(COLOR c, CASTLING_SIDE side) {
    if(castlings[board::_castling_index(c, side)] >= get_current_ply()) {
      castlings[board::_castling_index(c, side)] = get_current_ply();
    }
  }

  INLINE piece_bitboard_t get_castlings_mask() const {
    piece_bitboard_t mask = 0x00;
    if(is_castling(WHITE, KING_SIDE))mask|=0x40ULL;
    if(is_castling(WHITE, QUEEN_SIDE))mask|=0x04ULL;
    if(is_castling(BLACK, KING_SIDE))mask|=0x40ULL<<(board::SIZE-board::LEN);
    if(is_castling(BLACK, QUEEN_SIDE))mask|=0x04ULL<<(board::SIZE-board::LEN);
    return mask;
  }

  INLINE pos_t get_castlings_compressed() const {
    return fen::compress_castlings(get_castlings_mask());
  }

  void update_castlings(pos_t i, pos_t j) {
    const COLOR c = self[i].color;
    const pos_t shift = (c == WHITE) ? 0 : board::SIZE-board::LEN;
    if(self[i].value == KING) {
      unset_castling(c, KING_SIDE);
      unset_castling(c, QUEEN_SIDE);
    } else if(is_castling(c, QUEEN_SIDE) && self[i].value == ROOK && i == board::_pos(A, 1) + shift) {
      unset_castling(c, QUEEN_SIDE);
    } else if(is_castling(c, KING_SIDE) && self[i].value == ROOK && i == board::_pos(H, 1) + shift) {
      unset_castling(c, KING_SIDE);
    }

    const pos_t antishift = (c == BLACK) ? 0 : board::SIZE-board::LEN;
    if(is_castling(enemy_of(c), QUEEN_SIDE) && self[j].value == ROOK && j == board::_pos(A, 1) + antishift) {
      unset_castling(enemy_of(c), QUEEN_SIDE);
    } else if(is_castling(enemy_of(c), KING_SIDE) && self[j].value == ROOK && j == board::_pos(H, 1) + antishift) {
      unset_castling(enemy_of(c), KING_SIDE);
    }
  }

  ALWAYS_UNROLL INLINE void restore_castlings() {
    for(pos_t cstl = 0; cstl < castlings.size(); ++cstl) {
      if(castlings[cstl] > get_current_ply()) {
        castlings[cstl] = board::nocastlings;
      }
    }
  }

  zobrist::key_t zb_hash() const {
    zobrist::key_t zb = 0x00;
    for(pos_t i = 0; i < board::NO_PIECE_INDICES - 1; ++i) {
      bitmask::foreach(pieces[i].mask, [&](pos_t pos) mutable -> void {
        zb ^= zobrist::rnd_hashes[zobrist::rnd_start_piecepos + board::SIZE * i + pos];
      });
    }
    if(is_castling(WHITE,QUEEN_SIDE))zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 0];
    if(is_castling(WHITE,KING_SIDE) )zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 1];
    if(is_castling(BLACK,QUEEN_SIDE))zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 2];
    if(is_castling(BLACK,KING_SIDE) )zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 3];
    if(enpassant_trace() != event::enpassantnotrace) {
      zb ^= zobrist::rnd_hashes[zobrist::rnd_start_enpassant + board::_x(enpassant_trace())];
    }
    if(activePlayer() == BLACK) {
      zb ^= zobrist::rnd_hashes[zobrist::rnd_start_moveside];
    }
    return zb;
  }

  INLINE move_index_t get_current_ply() const {
    return history.size();
  }

  INLINE bool check_valid_move(pos_t i, pos_t j) const {
    return i == (i & board::MOVEMASK) &&
           self[i].color == activePlayer() &&
           self.get_moves_from(i) & (1ULL << (j & board::MOVEMASK));
  }

  void act_event(event_t ev) {
    history.emplace_back(ev);
    const uint8_t marker = event::extract_marker(ev);
    assert(fen::decompress_castlings(fen::compress_castlings(get_castlings_mask())) == get_castlings_mask());
    backup_state_on_event();
    switch(marker) {
      case event::NULLMOVE_MARKER:break;
      case event::BASIC_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m);
          const pos_t j = bitmask::second(m);
          const pos_t killwhat = event::extract_piece_ind(ev);
          const pos_t new_enpassant = event::extract_pos(ev);
          set_enpassant(new_enpassant);
          if(killwhat != event::killnothing || self[i].value == PAWN) {
            update_halfmoves();
          }
          update_castlings(i, j);
          move_pos(i, j);
          update_state_pos(i);
          update_state_pos(j);
        }
      break;
      case event::CASTLING_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          const move_t r_m = event::extract_move(ev);
          const pos_t r_i = bitmask::first(r_m),
                      r_j = bitmask::second(r_m);
          update_castlings(i, j);
          move_pos(i, j);
          update_state_pos(i);
          update_state_pos(j);
          move_pos(r_i, r_j);
          update_state_pos(r_i);
          update_state_pos(r_j);
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          const pos_t killwhere = event::extract_pos(ev);
          put_pos(killwhere, self.get_piece(EMPTY));
          update_state_pos(killwhere);
          move_pos(i, j);
          update_state_pos(i);
          update_state_pos(j);
          update_halfmoves();
        }
      break;
      case event::PROMOTION_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      to_byte = bitmask::second(m);
            const pos_t j = to_byte & board::MOVEMASK;
            const PIECE becomewhat = board::get_promotion_as(to_byte);
          const pos_t killwhat = event::extract_piece_ind(ev);
          auto new_enpassant = event::extract_pos(ev);
          update_castlings(i, j);
          put_pos(i, self.get_piece(EMPTY));
          put_pos(j, self.pieces[Piece::get_piece_index(becomewhat, activePlayer())]);
          update_state_pos(i);
          update_state_pos(j);
          update_halfmoves();
        }
      break;
    }
    activePlayer_ = enemy_of(activePlayer());
    update_state_on_event();
  }

  INLINE event_t last_event() const {
    if(history.empty())return 0x00;
    return history.back();
  }

  void unact_event() {
    if(history.empty())return;
    event_t ev = history.back();
    const uint8_t marker = event::extract_marker(ev);
    switch(marker) {
      case event::NULLMOVE_MARKER:break;
      case event::BASIC_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          const pos_t killwhat = event::extract_piece_ind(ev);
          auto new_enpassant = event::extract_byte(ev);
          move_pos(j, i);
          //update_state_pos(i);
          if(killwhat != event::killnothing) {
            put_pos(j, self.pieces[killwhat]);
          }
          //update_state_pos(j);
          auto enpassant_trace = event::extract_byte(ev);
        }
      break;
      case event::CASTLING_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          const move_t r_m = event::extract_move(ev);
          const pos_t r_i = bitmask::first(r_m),
                      r_j = bitmask::second(r_m);
          move_pos(j, i);
          //update_state_pos(j);
          //update_state_pos(i);
          move_pos(r_j, r_i);
          //update_state_pos(r_j);
          //update_state_pos(r_i);
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          const pos_t killwhere = event::extract_pos(ev);
          put_pos(killwhere, self.get_piece(PAWN, enemy_of(self[j].color)));
          //update_state_pos(killwhere);
          move_pos(j, i);
          //update_state_pos(j);
          //update_state_pos(i);
        }
      break;
      case event::PROMOTION_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m) & board::MOVEMASK;
          const pos_t killwhat = event::extract_piece_ind(ev);
          auto new_enpassant = event::extract_byte(ev);

          const COLOR c = self[j].color;
          if(killwhat != event::killnothing) {
            put_pos(j, self.pieces[killwhat]);
          } else {
            put_pos(j, self.get_piece(EMPTY));
          }
          //update_state_pos(j);
          put_pos(i, self.get_piece(PAWN, c));
          //update_state_pos(i);
        }
      break;
    }
    activePlayer_ = enemy_of(activePlayer());
    history.pop_back();
    restore_enpassants();
    restore_castlings();
    restore_halfmoves();
    restore_state_on_event();
  }

  INLINE event_t get_move_event(move_t m) const {
    if(m == board::nomove) {
      return event::noevent;
    }
    return get_move_event(bitmask::first(m), bitmask::second(m));
  }

  event_t get_move_event(pos_t i, pos_t j) const {
    event_t ev = 0x00;
    const pos_t promote_as = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    if(is_castling_move(i, j)) {
      ev = ev_castle(i, j);
    } else if(is_enpassant_take_move(i, j)) {
      ev = ev_take_enpassant(i, j);
    } else if(is_promotion_move(i, j)) {
      ev = ev_promotion(i, j | promote_as);
    } else {
      ev = ev_basic(i, j);
    }
    return ev;
  }

  INLINE void make_move(move_t m) {
    act_event(get_move_event(m));
  }

  INLINE void make_move(pos_t i, pos_t j) {
    assert(check_valid_move(i, j));
    make_move(bitmask::_pos_pair(i, j));
  }

  INLINE void retract_move() {
    unact_event();
  }

  INLINE auto move_guard(move_t m) {
    return make_move_guard(self, m);
  }

  INLINE auto recursive_move_guard() {
    return make_recursive_move_guard(self);
  }

  INLINE bool check_valid_move(move_t m) const {
    return check_valid_move(bitmask::first(m), bitmask::second(m));
  }

  template <typename C>
  INLINE bool check_valid_sequence(const C &s) const {
    auto rec = self.recursive_move_guard();
    for(const move_t &m : s) {
      if(!check_valid_move(m)) {
        return false;
      }
      rec.guard(m);
    }
    return true;
  }

  void init_update_state() {
    init_state_piece_positions();
    init_state_attacks();
    init_state_checkline();
    init_state_moves();
    history.reserve(100);
    halfmoves.reserve(bitmask::count_bits(get_piece_positions(BOTH)) - 2);
    enpassants.reserve(16);
    // memoization
    state_hist_attacks.reserve(100);
    state_hist_moves.reserve(100);
    state_hist_pins.reserve(100);
    state_hist_pins_rays.reserve(100);
  }

  INLINE void update_state_pos(pos_t pos) {
    update_state_attacks_pos(pos);
  }

  void backup_state_on_event() {
    const COLOR c = activePlayer();
    state_hist_attacks.emplace_back(state_attacks);
    state_hist_moves.emplace_back(state_moves);
    state_hist_pins.emplace_back(state_pins[c]);
    state_hist_pins_rays.emplace_back(state_pins_rays);
  }

  using board_mailbox_t = std::array <piece_bitboard_t, board::SIZE>;
  std::vector<board_mailbox_t> state_hist_attacks;
  std::vector<board_mailbox_t> state_hist_moves;
  void update_state_on_event() {
    init_state_piece_positions();
    update_state_checkline();
    init_state_moves();
  }

  void restore_state_on_event() {
    const COLOR c = activePlayer();
    init_state_piece_positions();
    state_attacks = state_hist_attacks.back();
    state_hist_attacks.pop_back();
    update_state_checkline();
    state_moves = state_hist_moves.back();
    state_hist_moves.pop_back();
    state_pins[c] = state_hist_pins.back();
    state_hist_pins.pop_back();
    state_pins_rays = state_hist_pins_rays.back();
    state_hist_pins_rays.pop_back();
  }

  std::array<piece_bitboard_t, NO_COLORS> state_piece_positions = {0x00, 0x00};
  ALWAYS_UNROLL INLINE piece_bitboard_t get_piece_positions(COLOR c) const {
    assert(c != NEUTRAL);
    if(c==BOTH)return ~get_piece(EMPTY).mask;
    piece_bitboard_t mask = 0ULL;
    for(PIECE p : {PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING}) {
      mask|=self.get_piece(p, c).mask;
    }
    return mask;
  }

  void init_state_piece_positions() {
    state_piece_positions[WHITE] = get_piece_positions(WHITE);
    state_piece_positions[BLACK] = get_piece_positions(BLACK);
  }

  std::array <piece_bitboard_t, board::SIZE> state_attacks = {0x00};
  ALWAYS_UNROLL void init_state_attacks() {
    for(auto&a:state_attacks)a=0ULL;
    for(COLOR c : {WHITE, BLACK}) {
      const piece_bitboard_t occupied = get_piece_positions(BOTH) & ~get_piece(KING, enemy_of(c)).mask;
      for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
        get_piece(p,c).foreach([&](pos_t pos) mutable noexcept -> void {
          state_attacks[pos] |= get_piece(p,c).get_attack(pos,occupied);
        });
      }
    }
  }

  INLINE piece_bitboard_t get_attacks_from(pos_t pos) const { return state_attacks[pos]; }

  INLINE piece_bitboard_t get_sliding_attacks_to(pos_t j, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return get_attacks_to(j,WHITE)|get_attacks_to(j,BLACK);
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    return (Attacks<BISHOPM>::get_attacks(j,occupied) & (get_piece(BISHOP,c).mask|get_piece(QUEEN,c).mask))
        | (Attacks<ROOKM>::get_attacks(j,occupied) & (get_piece(ROOK,c).mask|get_piece(QUEEN,c).mask));
  }

  INLINE piece_bitboard_t get_pawn_attacks_to(pos_t j, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return get_attacks_to(j,WHITE)|get_attacks_to(j,BLACK);
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    return get_piece(PAWN,enemy_of(c)).get_attack(j,occupied) & get_piece(PAWN,c).mask;
  }

  INLINE piece_bitboard_t get_attacks_to(pos_t j, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return get_attacks_to(j,WHITE)|get_attacks_to(j,BLACK);
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    return get_sliding_attacks_to(j, c)
        | (Attacks<KNIGHTM>::get_attacks(j) & get_piece(KNIGHT,c).mask)
        | (Attacks<KINGM>::get_attacks(j,occupied) & get_piece(KING,c).mask)
        | get_pawn_attacks_to(j, c);
  }

  INLINE piece_bitboard_t get_attack_counts_to(pos_t j, COLOR c=NEUTRAL) const {
    return bitmask::count_bits(get_attacks_to(j,c));
  }

  void update_state_attacks_pos(pos_t pos) {
    const piece_bitboard_t affected = (1ULL << pos) | get_sliding_attacks_to(pos, BOTH);
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    bitmask::foreach(affected, [&](pos_t i) mutable -> void {
      state_attacks[i] = self[i].get_attack(i, occupied & ~get_piece(KING,enemy_of(self[i].color)).mask);
    });
  }

  ALWAYS_UNROLL INLINE piece_bitboard_t get_attack_mask(COLOR c=NEUTRAL) const {
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

  std::array<piece_bitboard_t, NO_COLORS> state_checkline = {0x00,0x00};
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
  INLINE bool is_draw_halfmoves() const {
    return get_halfmoves() == 100 && !is_checkmate();
  }

  INLINE bool is_checkmate() const {
    const COLOR c = activePlayer();
    return (state_checkline[c] != ~0ULL) && (get_attacks_from(get_king_pos(c)) & ~get_attack_mask(enemy_of(c)));
  }

  INLINE bool can_move(COLOR c=NEUTRAL) const {
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

  INLINE bool is_draw_stalemate() const {
    const COLOR c = activePlayer();
    return !get_attacks_to(get_king_pos(c), enemy_of(c)) && !can_move(c);
  }

  inline bool is_draw_material() const {
    if(get_piece(QUEEN,BLACK).mask || get_piece(QUEEN,WHITE).mask
        || get_piece(ROOK,BLACK).mask || get_piece(ROOK,WHITE).mask)
    {
      return false;
    }
    const piece_bitboard_t occupied = get_piece_positions(BOTH);
    return (bitmask::count_bits(occupied) == 2) // two kings
        || (bitmask::count_bits(occupied) == 3 // material draw, kings and a bishop/knight
            && get_piece(KNIGHT,WHITE).size() + get_piece(BISHOP,WHITE).size()
               + get_piece(KNIGHT,BLACK).size() + get_piece(BISHOP,BLACK).size() == 1)
        || (bitmask::count_bits(occupied) == 4 // material draw, kings and a bishop/knight each
            && get_piece(KNIGHT,WHITE).size() + get_piece(BISHOP,WHITE).size() == 1
            && get_piece(KNIGHT,BLACK).size() + get_piece(BISHOP,BLACK).size() == 1);
  }

  inline bool is_draw() const {
    return is_draw_halfmoves() || is_draw_material() || is_draw_stalemate();
  }

  std::array <piece_bitboard_t, board::SIZE> state_moves = {0x00};
  ALWAYS_UNROLL void init_state_moves() {
    for(auto&m:state_moves)m=0x00;
    if(is_draw_halfmoves()||is_draw_material())return;
    const COLOR c = activePlayer();
    // maybe do this as a loop for incremental updates
    {
      const piece_bitboard_t friends = state_piece_positions[c],
                             foes  = state_piece_positions[enemy_of(c)],
                             attack_mask = get_attack_mask(enemy_of(c)),
                             pins = get_pins(c);
      const bool doublecheck = (state_checkline[c] == 0x00);
      for(pos_t p = 0; p < NO_PIECES; ++p) {
        if(doublecheck && p!=KING)continue;
        get_piece((PIECE)p,c).foreach([&](pos_t pos) mutable noexcept -> void {
          if(p==PAWN) {
            state_moves[pos] = get_attacks_from(pos) & (foes|(1ULL << enpassant_trace()));
            state_moves[pos] |= get_push_moves(c, pos, friends|foes);
            state_moves[pos] &= state_checkline[c];
          } else if(p==KING) {
            state_moves[pos] = get_attacks_from(pos) & ~friends & ~attack_mask;
            state_moves[pos] |= Moves<KINGM>::get_castling_moves(pos, friends|foes, attack_mask, get_castlings_mask(), c);
          } else {
            state_moves[pos] = get_attacks_from(pos) & ~friends;
            state_moves[pos] &= state_checkline[c];
          }
        });
      }
      init_state_moves_checkline_enpassant_takes(c);
      init_state_pins();
    }
  }

  void init_state_moves_checkline_enpassant_takes(COLOR c=NEUTRAL) {
    if(c==NEUTRAL)c=activePlayer();
    const pos_t etrace = enpassant_trace();
    const pos_t epawn = enpassant_pawn();
    if(etrace == event::enpassantnotrace)return;
    if(!bitmask::is_exp2(state_checkline[c]))return;
    const pos_t attacker = bitmask::log2_of_exp2(state_checkline[c]);
    if(epawn != attacker)return;
    const piece_bitboard_t apawns = get_pawn_attacks_to(etrace,c);
    if(!apawns)return;
    bitmask::foreach(apawns, [&](pos_t apawn) mutable -> void {
      state_moves[apawn] |= (1ULL << etrace);
    });
  }

  INLINE piece_bitboard_t get_pins(COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)c=activePlayer();
    if(c==BOTH)return get_pins(WHITE)|get_pins(BLACK);
    return state_pins[c];
  }

  template <typename F>
  inline void iter_attacking_xrays(pos_t j, F &&func, COLOR c=NEUTRAL) const {
    if(c==NEUTRAL)return;
    const piece_bitboard_t dstbit = 1ULL << j;
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t friends = state_piece_positions[c],
                           foes = state_piece_positions[ec];
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

  std::array<piece_bitboard_t, NO_COLORS> state_pins = {0x00, 0x00};
  std::array<piece_bitboard_t, board::SIZE> state_pins_rays;
  void init_state_pins() {
    for(auto &spr:state_pins_rays)spr=~0x00ULL;
    const COLOR c = activePlayer();
    // maybe do this as a loop for incremental updates
    {
      state_pins[c] = 0x00;
      const piece_bitboard_t friends = state_piece_positions[c] & ~get_piece(KING,c).mask;
      iter_attacking_xrays(get_king_pos(c), [&](pos_t i, piece_bitboard_t r) mutable -> void {
        const pos_t attacker = i;
        r |= 1ULL << attacker;
        r &= ~get_piece(KING,c).mask;
        const piece_bitboard_t pin = friends & r;
        if(pin) {
          assert(bitmask::is_exp2(pin));
          const pos_t pin_pos = bitmask::log2_of_exp2(pin);
          state_pins[c] |= pin;
          state_pins_rays[pin_pos] = r;
          // update moves
          state_moves[pin_pos] &= r;
        }
      }, c);
    }
    init_horizontal_enpassant_pin();
  }

  void init_horizontal_enpassant_pin() {
    if(enpassant_trace() == event::enpassantnotrace)return;
    // maybe do this as a loop for incremental updates
    {
      const COLOR c = activePlayer();
      if(state_checkline[c] != ~0ULL)return;
      const pos_t etrace = enpassant_trace();
      const pos_t epawn = enpassant_pawn();
      const pos_t kingpos = get_king_pos(c);
      const piece_bitboard_t h = bitmask::hline << (board::_y(kingpos) * board::LEN);
      if(!((h & get_piece(ROOK,enemy_of(c)).mask) || (h & get_piece(QUEEN, enemy_of(c)).mask)))return;
      if(!(h & (1ULL << epawn)))return;
      const piece_bitboard_t apawns = get_pawn_attacks_to(etrace,c);
      if(!apawns)return;
      bitmask::foreach(apawns, [&](pos_t apawn) mutable -> void {
        put_pos(apawn, get_piece(EMPTY));
        put_pos(epawn, get_piece(EMPTY));
        //printf("consider horizontal pin %hhu -> %hhu\n", apawn, etrace);
        if(get_sliding_attacks_to(kingpos,enemy_of(c))) {
          //printf("horizontal pin disable %hhu -> %hhu\n", apawn, etrace);
          state_pins[self[apawn].color] |= (1ULL << apawn);
          const piece_bitboard_t forbid = ~(1ULL << etrace);
          state_pins_rays[apawn] &= forbid;
          state_moves[apawn] &= forbid;
        }
        put_pos(apawn, pieces[get_piece(PAWN,c).piece_index]);
        put_pos(epawn, pieces[get_piece(PAWN,enemy_of(c)).piece_index]);
      });
    }
  }

  std::vector<piece_bitboard_t> state_hist_pins;
  std::vector<decltype(state_pins_rays)> state_hist_pins_rays;

  INLINE piece_bitboard_t get_moves_from(pos_t pos, COLOR c=NEUTRAL) const {
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

  std::string _move_str(move_t m) const {
    if(m==board::nomove)return "0000"s;
    return board::_move_str(m, self[bitmask::first(m)].value == PAWN);
  }

  template <typename C>
  std::vector<std::string> _line_str(const C &line, bool thorough=false) {
    std::vector<std::string> s;
    auto rec_guard = self.recursive_move_guard();
    for(const auto m : line) {
      if(m==board::nomove)break;
      s.emplace_back(_move_str(m));
      if(thorough) {
        rec_guard.guard(m);
      }
    }
    return s;
  }

  fen::FEN export_as_fen() const {
    fen::FEN f = {
      .board = std::string(),
      .active_player = activePlayer(),
      .castling_compressed = fen::compress_castlings(get_castlings_mask()),
      .enpassant = enpassant_trace(),
      .halfmove_clock = get_halfmoves(),
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

bool Board::initialized = false;

fen::FEN fen::export_from_board(const Board &board) {
  return board.export_as_fen();
}
