#pragma once


#include <vector>
#include <list>
#include <bitset>

#include <String.hpp>
#include <Piece.hpp>
#include <FEN.hpp>
#include <Zobrist.hpp>

#include <MoveScope.hpp>
#include <MoveLine.hpp>


// board view of the game
class Board {
protected:
  static bool initialized_;
  COLOR activePlayer_;
  size_t current_ply_ = 0;
public:
  bool traditional;
  pos_t kcastlrook[NO_COLORS] = {0xff, 0xff};
  pos_t qcastlrook[NO_COLORS] = {0xff, 0xff};
  struct board_info {
    COLOR active_player = NEUTRAL;
    pos_t enpassant;
    pos_pair_t castlings;
    piece_bitboard_t whites, blacks, diag_slid, orth_slid, pawns;

    INLINE bool operator==(board_info other) const noexcept {
      return active_player == other.active_player &&
             pawns == other.pawns &&
             whites == other.whites && blacks == other.blacks &&
             diag_slid == other.diag_slid && orth_slid == other.orth_slid &&
             castlings == other.castlings &&
             enpassant == other.enpassant;
    }

    INLINE void unset() {
      active_player = NEUTRAL;
    }

    INLINE bool is_unset() const {
      return active_player == NEUTRAL;
    }

    INLINE pos_t pos_king(COLOR c) const {
      assert(c < NO_COLORS);
      if(c == WHITE) {
        return pos_t(pawns);
      } else {
        return pos_t(pawns >> (7 * board::LEN));
      }
    }

    INLINE piece_bitboard_t pawn_bits() const {
      return pawns & ((bitmask::full >> (board::LEN * 2)) << board::LEN);
    }
  };
public:
  Board &self = *this;
  std::vector<std::pair<ply_index_t, pos_t>> state_hist_enpassants;
  std::array<ply_index_t, 4> castlings = {board::nocastlings, board::nocastlings, board::nocastlings, board::nocastlings};
  std::vector<ply_index_t> state_hist_halfmoves;

  std::array<piece_bitboard_t, 2> bits = {0x00, 0x00};
  piece_bitboard_t
    bits_slid_diag=0x00, bits_slid_orth=0x00,
    bits_pawns=0x00;

  struct board_state {
    using board_mailbox_t = std::array<piece_bitboard_t, board::SIZE>;
    board_info info;
    board_mailbox_t attacks;
    std::array<piece_bitboard_t, NO_COLORS> checkline = {0x00,0x00};
    board_mailbox_t moves;
    std::array<piece_bitboard_t, NO_COLORS> pins;
  };
  board_state state;
  std::vector<board_state> state_hist;

  std::array<pos_t, 2> pos_king = {piece::uninitialized_pos, piece::uninitialized_pos};
  ply_index_t state_hist_repetitions = INT16_MAX;

  const std::array<Piece, 2*6+1>  pieces = {
    Piece(PAWN, WHITE), Piece(PAWN, BLACK),
    Piece(KNIGHT, WHITE), Piece(KNIGHT, BLACK),
    Piece(BISHOP, WHITE), Piece(BISHOP, BLACK),
    Piece(ROOK, WHITE), Piece(ROOK, BLACK),
    Piece(QUEEN, WHITE), Piece(QUEEN, BLACK),
    Piece(KING, WHITE), Piece(KING, BLACK),
    Piece(EMPTY, NEUTRAL)
  };
  explicit Board(const fen::FEN &f):
    activePlayer_(f.active_player),
    current_ply_(f.fullmove * 2 - (f.active_player == WHITE ? 1 : 0)),
    traditional(f.traditional)
  {
    state_hist_halfmoves.emplace_back(current_ply_ - f.halfmove_clock);
    if(!initialized_) {
      M42::init();
      zobrist::init();
      initialized_ = true;
    }
    for(pos_t i = 0; i < board::SIZE; ++i) {
      set_pos(i, get_piece(EMPTY));
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
//    std::cout << "castlings: " << std::bitset<16>(f.castlings) << std::endl;
    if(bitmask::first(f.castlings)) {
      const pos_t wcastlmask = bitmask::first(f.castlings);
      if(bitmask::log2_msb(wcastlmask) < board::_x(pos_king[WHITE])) {
        qcastlrook[WHITE] = bitmask::log2_msb(wcastlmask);
      } else {
        unset_castling(WHITE, QUEEN_SIDE);
      }
      if(bitmask::log2(wcastlmask) > board::_x(pos_king[WHITE])) {
        kcastlrook[WHITE] = bitmask::log2(wcastlmask);
      } else {
        unset_castling(WHITE, KING_SIDE);
      }
    } else {
      unset_castling(WHITE, KING_SIDE);
      unset_castling(WHITE, QUEEN_SIDE);
    }
    if(bitmask::second(f.castlings)) {
      const pos_t bcastlmask = bitmask::second(f.castlings);
      if(bitmask::log2_msb(bcastlmask) < board::_x(pos_king[BLACK])) {
        qcastlrook[BLACK] = bitmask::log2_msb(bcastlmask);
      } else {
        unset_castling(BLACK, QUEEN_SIDE);
      }
      if(bitmask::log2(bcastlmask) > board::_x(pos_king[BLACK])) {
        kcastlrook[BLACK] = bitmask::log2(bcastlmask);
      } else {
        unset_castling(BLACK, KING_SIDE);
      }
    } else {
      unset_castling(BLACK, KING_SIDE);
      unset_castling(BLACK, QUEEN_SIDE);
    }
//    str::print(
//      is_castling(WHITE, KING_SIDE),
//      is_castling(WHITE, QUEEN_SIDE),
//      is_castling(BLACK, KING_SIDE),
//      is_castling(BLACK, QUEEN_SIDE)
//    );
    set_enpassant(f.enpassant);
    init_update_state();
  }

  INLINE constexpr COLOR activePlayer() const {
    return activePlayer_;
  }

  INLINE const Piece operator[](pos_t ind) const {
    assert(ind <= board::MOVEMASK);
    const piece_bitboard_t ind_mask = piece::pos_mask(ind);
    const COLOR c = self.color_at_pos(ind);
    if(c == NEUTRAL) {
      return get_piece(EMPTY);
    }
    if(bits_pawns & ind_mask) {
      return get_piece(PAWN, c);
    } else if(bits_slid_diag & bits_slid_orth & ind_mask) {
      return get_piece(QUEEN, c);
    } else if(bits_slid_diag & ind_mask) {
      return get_piece(BISHOP, c);
    } else if(bits_slid_orth & ind_mask) {
      return get_piece(ROOK, c);
    } else if(pos_king[c] == ind) {
      return get_piece(KING, c);
    } else {
      return get_piece(KNIGHT, c);
    }
    abort();
  }

  INLINE bool empty_at_pos(pos_t ind) {
    assert(ind <= board::MOVEMASK);
    return !piece::is_set(bits[WHITE]|bits[BLACK], ind);
  }

  INLINE COLOR color_at_pos(pos_t ind) const {
    assert(ind <= board::MOVEMASK);
    const piece_bitboard_t ind_mask = piece::pos_mask(ind);
    if(bits[WHITE] & ind_mask) {
      return WHITE;
    } else if(bits[BLACK] & ind_mask) {
      return BLACK;
    }
    return NEUTRAL;
  }

  INLINE const Piece get_piece(PIECE p=EMPTY, COLOR c=NEUTRAL) const {
    return pieces[Piece::get_piece_index(p, c)];
  }

  INLINE const Piece get_piece(const Piece p) const {
    return get_piece(p.value, p.color);
  }

  INLINE void update_state_info() {
    state.info = (board_info){
      .active_player=activePlayer(),
      .enpassant=enpassant_trace(),
      .castlings=get_castlings_rook_mask(),
      .whites=bits[WHITE],
      .blacks=bits[BLACK],
      .diag_slid=bits_slid_diag,
      .orth_slid=bits_slid_orth,
      .pawns=bits_pawns
             | (uint64_t(pos_king[BLACK]) << 7*board::LEN) \
             | uint64_t(pos_king[WHITE]),
    };
  }

  INLINE piece_bitboard_t get_king_bits() const {
    return piece::pos_mask(pos_king[WHITE]) | piece::pos_mask(pos_king[BLACK]);
  }

  INLINE piece_bitboard_t get_knight_bits() const {
    return (bits[WHITE] | bits[BLACK]) ^ (bits_slid_diag | bits_slid_orth | bits_pawns | get_king_bits());
  }

  INLINE void unset_pos(pos_t i) {
    const piece_bitboard_t i_mask = piece::pos_mask(i);
    if(bits[WHITE] & i_mask) {
      piece::unset_pos(bits[WHITE], i);
//      if(self[i].value == KING && pos_king[WHITE] == i) {
//        pos_king[WHITE] = piece::uninitialized_pos;
//      }
    } else if(bits[BLACK] & i_mask) {
      piece::unset_pos(bits[BLACK], i);
//      if(self[i].value == KING && pos_king[BLACK] == i) {
//        pos_king[BLACK] = piece::uninitialized_pos;
//      }
    }
    if(bits_pawns & i_mask) {
      piece::unset_pos(bits_pawns, i);
    } else {
      if(bits_slid_diag & i_mask) {
        piece::unset_pos(bits_slid_diag, i);
      }
      if(bits_slid_orth & i_mask) {
        piece::unset_pos(bits_slid_orth, i);
      }
    }
  }

  INLINE void set_pos(pos_t i, const Piece p) {
    if(p.color == WHITE) {
      piece::set_pos(bits[WHITE], i);
      if(p.value == KING) {
        pos_king[WHITE] = i;
      }
    } else if(p.color == BLACK) {
      piece::set_pos(bits[BLACK], i);
      if(p.value == KING) {
        pos_king[BLACK] = i;
      }
    }
    switch(p.value) {
      case BISHOP:piece::set_pos(bits_slid_diag, i);break;
      case ROOK:piece::set_pos(bits_slid_orth, i);break;
      case QUEEN:piece::set_pos(bits_slid_orth, i);
                 piece::set_pos(bits_slid_diag, i);break;
      case PAWN:piece::set_pos(bits_pawns, i);break;
      default:break;
    }
  }

  INLINE void put_pos(pos_t i, const Piece p) {
    unset_pos(i);
    set_pos(i, p);
  }

  INLINE bool is_castling_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    const COLOR c = self.color_at_pos(i);
    const pos_t castlrank = (c == WHITE) ? 1 : 8;
    if(traditional) {
      return (i == pos_king[c]) && (i == board::_pos(E, castlrank))
          && (j == board::_pos(C, castlrank) || j == board::_pos(G, castlrank));
    } else {
      return (i == pos_king[c]) && (bits_slid_orth & ~bits_slid_diag & bits[c] & piece::pos_mask(j));
    }
  }

  INLINE bool is_doublepush_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    if(~bits_pawns & piece::pos_mask(i))return false;
    const COLOR c = self.color_at_pos(i);
    return piece::is_pawn_double_push(c, i, j);
  }

  INLINE bool is_enpassant_take_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    return (bits_pawns & piece::pos_mask(i)) && j == enpassant_trace() && j != board::enpassantnotrace;
  }

  INLINE bool is_promotion_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    if(~bits_pawns & piece::pos_mask(i))return false;
    const COLOR c = self.color_at_pos(i);
    return piece::is_pawn_promotion_move(c, i, j);
  }

  INLINE bool is_naively_capture_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    const COLOR c = self.color_at_pos(i);
    return bits[enemy_of(c)] & piece::pos_mask(j);
  }

  INLINE bool is_naively_checking_move(pos_t i, pos_t j) const {
    j &= board::MOVEMASK;
    const COLOR c = activePlayer();
    const pos_t target = pos_king[enemy_of(c)];
    const piece_bitboard_t occupied = (bits[WHITE] | bits[BLACK]) & ~piece::pos_mask(i);
    if(bits_pawns & piece::pos_mask(i)) {
      return piece::get_pawn_attack(j, c) & piece::pos_mask(target);
    }
    if(bits_slid_diag & piece::pos_mask(i)) {
      const piece_bitboard_t mask = piece::get_sliding_diag_attack(j, occupied) & piece::pos_mask(target);
      if(mask)return true;
    }
    if(bits_slid_orth & piece::pos_mask(i)) {
      const piece_bitboard_t mask = piece::get_sliding_orth_attack(j, occupied) & piece::pos_mask(target);
      if(mask)return true;
    }
    if(get_knight_bits() & piece::pos_mask(i)) {
      return piece::get_knight_attack(j) & piece::pos_mask(target);
    }
    return false;
  }
//
  INLINE void move_pos_quiet(pos_t i, pos_t j) {
    assert(j <= board::MOVEMASK);
    const piece_bitboard_t i_mask = piece::pos_mask(i);
    const COLOR c = self.color_at_pos(i);
    piece::move_pos(bits[c], i, j);
    if(bits_pawns & i_mask) {
      piece::move_pos(bits_pawns, i, j);
    } else if(pos_king[c] == i) {
      pos_king[c] = j;
    } else {
      if(bits_slid_diag & i_mask) {
        piece::move_pos(bits_slid_diag, i, j);
      }
      if(bits_slid_orth & i_mask) {
        piece::move_pos(bits_slid_orth, i, j);
      }
    }
  }

  INLINE void move_pos(pos_t i, pos_t j) {
    assert(j <= board::MOVEMASK);
    unset_pos(j);
    move_pos_quiet(i, j);
  }

  INLINE pos_t enpassant_trace() const {
    if(state_hist_enpassants.empty())return board::enpassantnotrace;
    const auto [ply, e] = state_hist_enpassants.back();
    if(ply == get_current_ply()) {
      return e;
    }
    return board::enpassantnotrace;
  }

  INLINE pos_t enpassant_pawn() const {
    if(enpassant_trace() == board::enpassantnotrace)return 0xFF;
    const pos_t x = board::_x(enpassant_trace());
    return board::_y(enpassant_trace()) == 3-1 ? board::_pos(A+x, 4) : board::_pos(A+x, 5);
  }

  INLINE void set_enpassant(pos_t e) {
    if(e != board::enpassantnotrace) {
      state_hist_enpassants.emplace_back(get_current_ply(), e);
    }
  }

  INLINE pos_t get_halfmoves() const {
    return get_current_ply() - state_hist_halfmoves.back();
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

  INLINE pos_pair_t get_castlings_rook_mask() const {
    pos_t wcastl=0x00, bcastl=0x00;
    if(is_castling(WHITE, KING_SIDE)) wcastl|=piece::pos_mask(kcastlrook[WHITE]);
    if(is_castling(WHITE, QUEEN_SIDE))wcastl|=piece::pos_mask(qcastlrook[WHITE]);
    if(is_castling(BLACK, KING_SIDE)) bcastl|=piece::pos_mask(kcastlrook[BLACK]);
    if(is_castling(BLACK, QUEEN_SIDE))bcastl|=piece::pos_mask(qcastlrook[BLACK]);
    return bitmask::_pos_pair(wcastl, bcastl);
  }

  void update_castlings(pos_t i, pos_t j) {
    const COLOR c = self.color_at_pos(i);
    const piece_bitboard_t rooks = bits_slid_orth & ~bits_slid_diag;
    if(pos_king[c] == i) {
      unset_castling(c, KING_SIDE);
      unset_castling(c, QUEEN_SIDE);
    } else if(rooks & bits[c] & piece::pos_mask(i)) {
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      if(is_castling(c, QUEEN_SIDE) && i == board::_pos(qcastlrook[c], castlrank)) {
        unset_castling(c, QUEEN_SIDE);
      } else if(is_castling(c, KING_SIDE) && i == board::_pos(kcastlrook[c], castlrank)) {
        unset_castling(c, KING_SIDE);
      }
    }

    const COLOR ec = enemy_of(c);
    if(rooks & bits[ec] & piece::pos_mask(j)) {
      const pos_t ecastlrank = (ec == WHITE) ? 1 : 8;
      if(is_castling(ec, QUEEN_SIDE) && j == board::_pos(qcastlrook[ec], ecastlrank)) {
        unset_castling(ec, QUEEN_SIDE);
      } else if(is_castling(ec, KING_SIDE) && j == board::_pos(kcastlrook[ec], ecastlrank)) {
        unset_castling(ec, KING_SIDE);
      }
    }
  }

  piece_bitboard_t get_mask(const Piece p) const {
    const COLOR c = p.color;
    if(p.value == PAWN) {
      return bits_pawns & bits[c];
    } else if(p.value == KING) {
      return piece::pos_mask(pos_king[c]);
    }
    piece_bitboard_t mask = 0x00;
    if(p.value == BISHOP || p.value == QUEEN) {
      mask |= bits_slid_diag;
    }
    if(p.value == ROOK || p.value == QUEEN) {
      mask |= bits_slid_orth;
    }
    if(p.value == KNIGHT) {
      mask = get_knight_bits();
    }
    return mask & bits[c];
  }

  zobrist::key_t zb_hash() const {
    zobrist::key_t zb = 0x00;
    for(pos_t i = 0; i < board::NO_PIECE_INDICES - 1; ++i) {
      const Piece p = self.pieces[i];
      bitmask::foreach(get_mask(p), [&](pos_t pos) mutable -> void {
        zb ^= zobrist::rnd_hashes[zobrist::rnd_start_piecepos + board::SIZE * i + pos];
      });
    }
    if(is_castling(WHITE,QUEEN_SIDE))zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 0];
    if(is_castling(WHITE,KING_SIDE) )zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 1];
    if(is_castling(BLACK,QUEEN_SIDE))zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 2];
    if(is_castling(BLACK,KING_SIDE) )zb^=zobrist::rnd_hashes[zobrist::rnd_start_castlings + 3];
    if(enpassant_trace() != board::enpassantnotrace) {
      zb ^= zobrist::rnd_hashes[zobrist::rnd_start_enpassant + board::_x(enpassant_trace())];
    }
    if(activePlayer() == BLACK) {
      zb ^= zobrist::rnd_hashes[zobrist::rnd_start_moveside];
    }
    return zb;
  }

  INLINE ply_index_t get_current_ply() const {
    return current_ply_;
  }

  INLINE bool check_valid_move(pos_t i, pos_t j) const {
    return i == (i & board::MOVEMASK)
           && (bits[activePlayer()] & piece::pos_mask(i))
           && (state.moves[i] & piece::pos_mask(j & board::MOVEMASK));
  }

  virtual void _backup_on_event() {}
  virtual void _restore_on_event() {}
  virtual void _update_pos_change(pos_t i, pos_t j) {};
  virtual void _update_change() {}

  void make_move(pos_t i, pos_t j) {
    const move_t m = bitmask::_pos_pair(i, j);
    const pos_t promote_as = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    const bool is_castling = is_castling_move(i, j);
    const bool is_enpassant_take = is_enpassant_take_move(i, j);
    const pos_t epawn = enpassant_pawn();

    ++current_ply_;
    _backup_on_event();
    state_hist.emplace_back(state);
    assert(m == board::nomove || check_valid_move(m));
    if(m == board::nomove) {
    } else if(is_castling) {
      const piece_bitboard_t knights = get_knight_bits();
      const COLOR c = activePlayer();
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      pos_t k_j = j;
      if(!traditional) {
        if(j == board::_pos(qcastlrook[c], castlrank)) {
          k_j = board::_pos(C, castlrank);
        } else if(j == board::_pos(kcastlrook[c], castlrank)) {
          k_j = board::_pos(G, castlrank);
        } else {
          abort();
        }
      }
      const move_t rookmove = piece::get_king_castle_rook_move(c, i, k_j, qcastlrook[c], kcastlrook[c]);
      const pos_t r_i = bitmask::first(rookmove),
                  r_j = bitmask::second(rookmove);
      update_castlings(i, j);
      {
        if(k_j != r_i) {
          piece::move_pos(bits[c], i, k_j);
        }
        pos_king[c] = k_j;
      }
      update_state_attacks_pos(i);
      update_state_attacks_pos(k_j);
      _update_pos_change(i, k_j);
      {
        if(k_j != r_i) {
          piece::move_pos(bits[c], r_i, r_j);
        }
        piece::move_pos(bits_slid_orth, r_i, r_j);
      }
      update_state_attacks_pos(r_i);
      update_state_attacks_pos(r_j);
      _update_pos_change(r_i, r_j);
      const COLOR ec = enemy_of(c);
      assert((bits[c] | bits[ec]) == (bits_pawns | bits_slid_diag | bits_slid_orth | get_king_bits() | knights));
    } else if(is_enpassant_take) {
      const COLOR c = activePlayer();
      const pos_t killwhere = epawn;
      unset_pos(killwhere);
      update_state_attacks_pos(killwhere);
      {
        piece::move_pos(bits[c], i, j);
        piece::move_pos(bits_pawns, i, j);
      }
      update_state_attacks_pos(i);
      update_state_attacks_pos(j);
      _update_pos_change(i, j);
      state_hist_halfmoves.emplace_back(get_current_ply());
    } else if(is_promotion_move(i, j)) {
      const PIECE becomewhat = board::get_promotion_as(promote_as);
      update_castlings(i, j);
      unset_pos(i);
      put_pos(j, self.pieces[Piece::get_piece_index(becomewhat, activePlayer())]);
      update_state_attacks_pos(i);
      update_state_attacks_pos(j);
      _update_pos_change(i, j);
      state_hist_halfmoves.emplace_back(get_current_ply());
    } else {
      const bool killmove = !self.empty_at_pos(j);
      pos_t new_enpassant = board::enpassantnotrace;
      if(is_doublepush_move(i, j)) {
        const COLOR c = activePlayer();
        new_enpassant = piece::get_pawn_enpassant_trace(c, i, j);
      }
      if(killmove || bits_pawns & piece::pos_mask(i)) {
        state_hist_halfmoves.emplace_back(get_current_ply());
      }
      set_enpassant(new_enpassant);
      update_castlings(i, j);
      if(!killmove) {
        move_pos_quiet(i, j);
      } else {
        move_pos(i, j);
      }
      update_state_attacks_pos(i);
      update_state_attacks_pos(j);
      _update_pos_change(i, j);
    }
    activePlayer_ = enemy_of(activePlayer());
    init_state_moves();
    update_state_info();
    if(state_hist_repetitions > self.get_current_ply()) {
      update_state_repetitions();
    }
    _update_change();
  }

  INLINE void make_move(move_t m) {
    make_move(bitmask::first(m), bitmask::second(m));
  }

  void retract_move() {
    if(state_hist.empty())return;
    const board_info prev_info = state_hist.back().info;
    bits[WHITE] = prev_info.whites;
    bits[BLACK] = prev_info.blacks;
    bits_pawns = prev_info.pawn_bits();
    bits_slid_diag = prev_info.diag_slid;
    bits_slid_orth = prev_info.orth_slid;
    pos_king[WHITE] = prev_info.pos_king(WHITE);
    pos_king[BLACK] = prev_info.pos_king(BLACK);
    activePlayer_ = enemy_of(activePlayer());
    --current_ply_;
    //enpassants
    while(!state_hist_enpassants.empty() && state_hist_enpassants.back().first > get_current_ply()) {
      state_hist_enpassants.pop_back();
    }
    // castlings
    for(pos_t cstl = 0; cstl < castlings.size(); ++cstl) {
      if(castlings[cstl] > get_current_ply()) {
        castlings[cstl] = board::nocastlings;
      }
    }
    // halfmoves
    if(state_hist_halfmoves.size() > 1) {
      while(state_hist_halfmoves.back() > get_current_ply()) {
        state_hist_halfmoves.pop_back();
      }
    }
    // state
    state = state_hist.back();
    state_hist.pop_back();
    if(self.get_current_ply() < state_hist_repetitions) {
      state_hist_repetitions = INT16_MAX;
    }
    _restore_on_event();
  }

  INLINE auto move_scope(move_t m) {
    return make_move_scope(self, m);
  }

  INLINE auto mline_scope(move_t m, MoveLine &mline) {
    return make_mline_scope(self, m, mline);
  }

  INLINE auto recursive_move_scope() {
    return make_recursive_move_scope(self);
  }

  INLINE auto recursive_mline_scope(MoveLine &mline) {
    return make_recursive_mline_scope(self, mline);
  }

  INLINE bool check_valid_move(move_t m) const {
    return check_valid_move(bitmask::first(m), bitmask::second(m));
  }

  INLINE bool check_valid_sequence(const MoveLine &mline) {
    auto rec = self.recursive_move_scope();
    for(const move_t &m : mline) {
      if(!check_valid_move(m)) {
        return false;
      }
      rec.scope(m);
    }
    return true;
  }

  INLINE bool check_line_terminates(const MoveLine &mline) {
    auto rec = self.recursive_move_scope();
    for(const move_t &m : mline) {
      rec.scope(m);
    }
    return is_draw() || !can_move() || can_draw_repetition();
  }

  INLINE void init_update_state() {
    init_state_attacks();
    init_state_moves();
    state_hist_halfmoves.reserve(piece::size(bits[WHITE] | bits[BLACK]) - 2);
    state_hist_enpassants.reserve(16);
    state_hist.reserve(100);
    update_state_info();
  }

  using board_mailbox_t = std::array <piece_bitboard_t, board::SIZE>;
  ALWAYS_UNROLL void init_state_attacks() {
    for(auto&a:state.attacks)a=0ULL;
    for(COLOR c : {WHITE, BLACK}) {
      const piece_bitboard_t occupied = (bits[WHITE] | bits[BLACK]) & ~piece::pos_mask(pos_king[enemy_of(c)]);

      bitmask::foreach(bits_pawns & bits[c], [&](pos_t pos) mutable noexcept -> void {
        state.attacks[pos] |= piece::get_pawn_attack(pos,c);
      });
      bitmask::foreach(bits_slid_diag & bits[c], [&](pos_t pos) mutable noexcept -> void {
        state.attacks[pos] |= piece::get_sliding_diag_attack(pos,occupied);
      });
      bitmask::foreach(bits_slid_orth & bits[c], [&](pos_t pos) mutable noexcept -> void {
        state.attacks[pos] |= piece::get_sliding_orth_attack(pos,occupied);
      });
      state.attacks[pos_king[c]] |= piece::get_king_attack(pos_king[c]);
    }
    bitmask::foreach(get_knight_bits(), [&](pos_t pos) mutable noexcept -> void {
      state.attacks[pos] |= piece::get_knight_attack(pos);
    });
  }

  INLINE piece_bitboard_t get_sliding_diag_attacks_to(pos_t j, piece_bitboard_t occupied) const {
    return piece::get_sliding_diag_attack(j,occupied) & bits_slid_diag & occupied;
  }

  INLINE piece_bitboard_t get_sliding_orth_attacks_to(pos_t j, piece_bitboard_t occupied) const {
    return piece::get_sliding_orth_attack(j,occupied) & bits_slid_orth & occupied;
  }

  INLINE piece_bitboard_t get_sliding_attacks_to(pos_t j, piece_bitboard_t occupied) const {
    return get_sliding_diag_attacks_to(j, occupied) | get_sliding_orth_attacks_to(j, occupied);
  }

  INLINE piece_bitboard_t get_pawn_attacks_to(pos_t j, COLOR c) const {
    assert(c == BOTH || c < NO_COLORS);
    if(c==BOTH)return get_pawn_attacks_to(j, WHITE) | get_pawn_attacks_to(j, BLACK);
    return piece::get_pawn_attack(j,enemy_of(c)) & (bits_pawns & bits[c]);
  }

  INLINE piece_bitboard_t get_king_attacks_to(pos_t j) const {
    return piece::get_king_attack(j) & get_king_bits();
  }

  INLINE piece_bitboard_t get_attacks_to(pos_t j, COLOR c, piece_bitboard_t occupied) const {
    assert(c == BOTH || c < NO_COLORS);
    const piece_bitboard_t filt = (c == BOTH) ? occupied : (occupied & bits[c]);
    return (get_sliding_attacks_to(j, occupied)
      | (piece::get_knight_attack(j) & get_knight_bits())
      | get_king_attacks_to(j)
      | get_pawn_attacks_to(j, c)) & filt;
  }

  INLINE piece_bitboard_t get_attack_counts_to(pos_t j, COLOR c) const {
    return bitmask::count_bits(get_attacks_to(j,c,bits[WHITE]|bits[BLACK]));
  }

  void update_state_attacks_pos(pos_t pos) {
    const piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    const piece_bitboard_t affected = piece::pos_mask(pos) | get_sliding_attacks_to(pos, occupied);

    for(COLOR c : {WHITE, BLACK}) {
      bitmask::foreach(affected & bits[c], [&](pos_t pos) mutable noexcept -> void {
        const piece_bitboard_t posmask = piece::pos_mask(pos);
        const piece_bitboard_t occ_c = occupied & ~piece::pos_mask(pos_king[enemy_of(c)]);
        if(~occupied & posmask) {
          state.attacks[pos] = 0x00;
          return;
        } else if(bits_pawns & posmask) {
          state.attacks[pos] = piece::get_pawn_attack(pos, c);
        } else if(pos == pos_king[WHITE] || pos == pos_king[BLACK]) {
          state.attacks[pos] = piece::get_king_attack(pos);
        } else if(posmask & (bits_slid_diag | bits_slid_orth)) {
          state.attacks[pos] = 0x00;
          if(posmask & bits_slid_diag) {
            state.attacks[pos] |= piece::get_sliding_diag_attack(pos,occ_c);
          }
          if(posmask & bits_slid_orth) {
            state.attacks[pos] |= piece::get_sliding_orth_attack(pos,occ_c);
          }
        } else {
          state.attacks[pos] = piece::get_knight_attack(pos);
        }
      });
    }
  }

  INLINE piece_bitboard_t get_attack_mask(COLOR c) const {
    assert(c < NO_COLORS);
    const piece_bitboard_t occupied = (bits[WHITE] | bits[BLACK]) & ~piece::pos_mask(pos_king[enemy_of(c)]);
    piece_bitboard_t mask = 0x00;
    mask |= piece::get_pawn_attacks(bits_pawns & bits[c], c);
    mask |= piece::get_sliding_diag_attacks(bits_slid_diag & bits[c], occupied);
    mask |= piece::get_sliding_orth_attacks(bits_slid_orth & bits[c], occupied);
    mask |= piece::get_knight_attacks(get_knight_bits() & bits[c]);
    mask |= piece::get_king_attack(pos_king[c]);
    return mask;
  }

  void init_state_checkline(COLOR c) {
    assert(c < NO_COLORS);
    const piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    const piece_bitboard_t attackers = get_attacks_to(pos_king[c], enemy_of(c), occupied);
    if(!attackers) {
      state.checkline[c] = ~0x00ULL;
      return;
    } else if(!bitmask::is_exp2(attackers)) {
      state.checkline[c] = 0x00ULL;
      return;
    }
    const pos_t attacker = bitmask::log2_of_exp2(attackers);
    state.checkline[c] = 0x00;
    if(bits_slid_diag & attackers) {
      state.checkline[c] |= piece::get_sliding_diag_attacking_ray(pos_king[c], attacker, occupied & ~piece::pos_mask(pos_king[c]));
    }
    if(bits_slid_orth & attackers) {
      state.checkline[c] |= piece::get_sliding_orth_attacking_ray(pos_king[c], attacker, occupied & ~piece::pos_mask(pos_king[c]));
    }
    state.checkline[c] |= attackers;
  }

  // fifty-halfmoves-draw
  INLINE bool is_draw_halfmoves() const {
    return get_halfmoves() == 100;// && !is_checkmate();
  }

  INLINE bool is_checkmate() const {
    const COLOR c = activePlayer();
    if(state.checkline[c] == bitmask::full || (state.attacks[pos_king[c]] & ~bits[c] & ~get_attack_mask(enemy_of(c)))) {
      return false;
    }
    return !is_draw() && !can_move();
  }

  INLINE bool can_move() const {
    const COLOR c = activePlayer();
    bool canmove = false;
    bitmask::foreach_early_stop(bits[c], [&](pos_t pos) mutable -> bool {
      if(state.moves[pos])canmove=true;
      return !canmove;
    });
    return canmove;
  }

  INLINE bool is_draw_stalemate() const {
    const COLOR c = activePlayer();
    return !get_attacks_to(pos_king[c], enemy_of(c), bits[WHITE]|bits[BLACK]) && !can_move();
  }

  INLINE bool is_draw_material() const {
    if(bits_slid_orth)return false;
    const size_t no_pieces = piece::size(bits[WHITE] | bits[BLACK]);
    const piece_bitboard_t bishops = bits_slid_diag & ~bits_slid_orth,
                           knights = get_knight_bits();
    const piece_bitboard_t light_pieces = knights | bishops;
    return (no_pieces == 2)
        || (no_pieces == 3 && piece::size(light_pieces) == 1)
        || (no_pieces == 4
            && piece::size(light_pieces & bits[WHITE]) == 1
            && piece::size(light_pieces & bits[BLACK]) == 1);
  }

  INLINE bool is_draw() const {
    return is_draw_halfmoves() || is_draw_material() || is_draw_stalemate();
  }

  INLINE void update_state_repetitions() {
    if(state_hist_repetitions <= self.get_current_ply()) {
      state_hist_repetitions = get_current_ply();
      return;
    }
    int repetitions = 1;
    for(ply_index_t i = 0; i < std::min<ply_index_t>(state_hist.size(), get_halfmoves()); ++i) {
      if(state.info == state_hist[state_hist.size() - i - 1].info) {
        ++repetitions;
        if(repetitions >= 3) {
          state_hist_repetitions = get_current_ply();
          return;
        }
      }
    }
    return;
  }

  INLINE bool can_draw_repetition() const {
    return state_hist_repetitions <= self.get_current_ply();
  }

  void init_state_moves() {
    for(auto&m:state.moves)m=0x00;
    if(is_draw_halfmoves()||is_draw_material())return;
    // for efficiency set c = activePlayer()
    for(const COLOR c : {WHITE, BLACK}) {
//    const COLOR c = activePlayer(); {
      init_state_checkline(c);
      const COLOR ec = enemy_of(c);
      const piece_bitboard_t friends = bits[c], foes = bits[ec],
                             attack_mask = get_attack_mask(ec);
      const bool doublecheck = (state.checkline[c] == 0x00);
      if(!doublecheck) {
        const pos_t etrace = enpassant_trace();
        piece_bitboard_t foes_pawn = foes;
        if(etrace != board::enpassantnotrace && c == activePlayer()) {
          foes_pawn |= piece::pos_mask(etrace);
        }
        bitmask::foreach(bits_pawns & bits[c], [&](pos_t pos) mutable noexcept -> void {
          state.moves[pos] = state.attacks[pos] & foes_pawn;
          state.moves[pos] |= piece::get_pawn_push_moves(c, pos, friends|foes);
          state.moves[pos] &= state.checkline[c];
        });
        bitmask::foreach(bits[c] & ~(bits_pawns | get_king_bits()), [&](pos_t pos) mutable noexcept -> void {
          state.moves[pos] = state.attacks[pos] & ~friends & state.checkline[c];
        });
      }
      state.moves[pos_king[c]] = state.attacks[pos_king[c]] & ~friends & ~attack_mask;
      if(state.checkline[c] == bitmask::full && (is_castling(c, QUEEN_SIDE) || is_castling(c, KING_SIDE))) {
        state.moves[pos_king[c]] |= piece::get_king_castling_moves(c, pos_king[c], friends|foes, attack_mask,
                                                                   is_castling(c, QUEEN_SIDE), is_castling(c, KING_SIDE),
                                                                   qcastlrook[c], kcastlrook[c], traditional);
      }
    }
    init_state_moves_checkline_enpassant_takes();
    init_state_pins();
    if(!traditional) {
      for(const COLOR c : {WHITE, BLACK}) {
//      const COLOR c = activePlayer(); {
        state.moves[pos_king[c]] &= ~state.pins[c];
      }
    }
  }

  void init_state_moves_checkline_enpassant_takes() {
    const COLOR c = activePlayer();
    const pos_t etrace = enpassant_trace();
    const pos_t epawn = enpassant_pawn();
    if(etrace == board::enpassantnotrace)return;
    if(!bitmask::is_exp2(state.checkline[c]))return;
    const pos_t attacker = bitmask::log2_of_exp2(state.checkline[c]);
    if(epawn != attacker)return;
    const piece_bitboard_t apawns = get_pawn_attacks_to(etrace,c);
    if(!apawns)return;
    bitmask::foreach(apawns, [&](pos_t apawn) mutable noexcept -> void {
      state.moves[apawn] |= piece::pos_mask(etrace);
    });
  }

  template <typename F>
  INLINE void iter_attacking_xrays(pos_t j, F &&func, COLOR c) const {
    assert(c < NO_COLORS);
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t dstbit = piece::pos_mask(j);
    const piece_bitboard_t friends = bits[c],
                           foes = bits[ec];
    const piece_bitboard_t orth_xray = piece::get_sliding_orth_xray_attack(j, foes, friends) & foes & bits_slid_orth;
    bitmask::foreach(orth_xray, [&](pos_t i) mutable -> void {
      // includes dstbit in the xray
      const piece_bitboard_t r = piece::get_sliding_orth_attacking_xray(i,j,foes|dstbit,friends);
      func(i, r);
    });
    const piece_bitboard_t diag_xray = piece::get_sliding_diag_xray_attack(j, foes, friends) & foes & bits_slid_diag;
      // includes dstbit in the xray
    bitmask::foreach(diag_xray, [&](pos_t i) mutable -> void {
      const piece_bitboard_t r = piece::get_sliding_diag_attacking_xray(i,j,foes|dstbit,friends);
      func(i, r);
    });
  }

  void init_state_pins() {
    // for efficiency can avoid the loop
    for(COLOR c : {WHITE, BLACK}) {
//    const COLOR c = activePlayer(); {
      state.pins[c] = 0x00ULL;
      const piece_bitboard_t kingmask = piece::pos_mask(pos_king[c]);
      iter_attacking_xrays(pos_king[c], [&](pos_t attacker, piece_bitboard_t r) mutable -> void {
        // include attacker into the ray
        r |= piece::pos_mask(attacker);
        // exclude the king from it
        r &= ~kingmask;
        const piece_bitboard_t pin = bits[c] & r;
        if(pin) {
          assert(bitmask::is_exp2(pin));
          const pos_t pin_pos = bitmask::log2_of_exp2(pin);
          state.pins[c] |= pin;
          // update moves
          state.moves[pin_pos] &= r;
        }
      }, c);
    }
    init_horizontal_enpassant_pin();
  }

  void init_horizontal_enpassant_pin() {
    if(enpassant_trace() == board::enpassantnotrace)return;
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    const pos_t r = (c == WHITE) ? -1+5 : -1+4;
    if(
        state.checkline[c] != bitmask::full
        || !(piece::rank_mask(r) & bits[ec] & bits_slid_orth & ~bits_slid_diag))
    {
      return;
    }
    const pos_t epawn = enpassant_pawn(),
                etrace = enpassant_trace();
    const piece_bitboard_t apawns = get_pawn_attacks_to(etrace,c);
    if(!bitmask::is_exp2(apawns))return;
    const pos_t apawn = bitmask::log2_of_exp2(apawns);
    piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    occupied &= ~(piece::pos_mask(apawn) | piece::pos_mask(epawn));
    //printf("consider horizontal pin %hhu -> %hhu\n", apawn, etrace);
    if(get_sliding_orth_attacks_to(pos_king[c],occupied) & bits[ec]) {
      //printf("horizontal pin disable %hhu -> %hhu\n", apawn, etrace);
      state.pins[c] |= piece::pos_mask(apawn);
      state.moves[apawn] &= ~piece::pos_mask(etrace);
    }
  }

  NEVER_INLINE void print() const {
    for(pos_t i = board::LEN; i > 0; --i) {
      for(pos_t j = 0; j < board::LEN; ++j) {
        const Piece p = self[(i-1) * board::LEN + j];
        std::cout << p.str() << " ";
      }
      std::cout << std::endl;
    }
  }

  NEVER_INLINE std::string _move_str(move_t m) const {
    if(m==board::nomove)return "0000"s;
    return board::_move_str(m, bits_pawns & piece::pos_mask(bitmask::first(m)));
  }

  NEVER_INLINE std::vector<std::string> _line_str(const MoveLine &line, bool thorough=false) {
    std::vector<std::string> s;
    if(thorough)assert(check_valid_sequence(line));
    auto rec_mscope = self.recursive_move_scope();
    for(const auto m : line) {
      if(m==board::nomove)break;
      s.emplace_back(_move_str(m));
      if(thorough) {
        rec_mscope.scope(m);
      }
    }
    return s;
  }

  NEVER_INLINE std::string _line_str_full(const MoveLine &line, bool thorough_fut=false) {
    std::string s = "["s + str::join(_line_str(line.get_past(), false), " "s) + "]"s;
    if(!line.empty()) {
      s += " "s + str::join(_line_str(line.get_future(), thorough_fut), " "s);
    }
    return s;
  }

  board_info get_info_from_line(MoveLine &mline) {
    assert(check_valid_sequence(mline));
    auto rec = self.recursive_move_scope();
    for(const move_t &m : mline) {
      rec.scope(m);
    }
    return state.info;
  }

  fen::FEN export_as_fen() const {
    fen::FEN f = {
      .board=""s,
      .active_player=activePlayer(),
      .castlings=get_castlings_rook_mask(),
      .enpassant=enpassant_trace(),
      .halfmove_clock=get_halfmoves(),
      .fullmove=ply_index_t(((get_current_ply() - 1) / 2) + 1),
      .traditional=traditional
    };
    for(pos_t y = 0; y < board::LEN; ++y) {
      for(pos_t x = 0; x < board::LEN; ++x) {
        const pos_t ind = board::_pos(A+x, 8-y);
        if(self.empty_at_pos(ind)) {
          f.board += ' ';
        } else {
          f.board += self[ind].str();
        }
      }
    }
    return f;
  }
};

bool Board::initialized_ = false;

fen::FEN fen::export_from_board(const Board &board) {
  return board.export_as_fen();
}
