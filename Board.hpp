#pragma once


#include <vector>
#include <list>

#include <String.hpp>
#include <Piece.hpp>
#include <Event.hpp>
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

  struct board_info {
    COLOR active_player = NEUTRAL;
    pos_t enpassant_castlings;
    piece_bitboard_t whites, blacks, diag_slid, orth_slid, pawns;

    INLINE bool operator==(board_info other) const noexcept {
      return active_player == other.active_player &&
             pawns == other.pawns &&
             whites == other.whites && blacks == other.blacks &&
             diag_slid == other.diag_slid && orth_slid == other.orth_slid &&
             enpassant_castlings == other.enpassant_castlings;
    }

    INLINE void unset() {
      active_player = NEUTRAL;
    }

    INLINE bool is_unset() const {
      return active_player == NEUTRAL;
    }

    static board_info import_dbg(const std::string &s) {
      auto vs = str::split(s, "|"s);
      return (board_info){
        .active_player=COLOR(std::stoi(vs[6])),
        .enpassant_castlings=pos_t(std::stoi(vs[0])),
        .whites=std::stoull(vs[2]),
        .blacks=std::stoull(vs[3]),
        .diag_slid=std::stoull(vs[4]),
        .orth_slid=std::stoull(vs[5]),
        .pawns=std::stoull(vs[1]),
      };
    }

    std::string export_dbg() const {
      std::string s = std::to_string(enpassant_castlings) + "|"s;
      for(uint64_t it : {pawns, whites, blacks, diag_slid, orth_slid}) {
        s += std::to_string(it) + "|"s;
      }
      s += std::to_string(active_player);
      return s;
    }

    INLINE pos_t pos_king(COLOR c) const {
      assert(c == WHITE || c == BLACK);
      if(c == WHITE) {
        return pos_t(pawns);
      } else {
        return pos_t(pawns >> (7 * board::LEN));
      }
    }

    INLINE piece_bitboard_t pawn_bits() const {
      return pawns & ((bitmask::full >> (board::LEN * 2)) << board::LEN);
    }

    INLINE pos_t etrace() const {
      return enpassant_castlings >> 4;
    }
  };
public:
  Board &self = *this;
  std::vector<std::pair<ply_index_t, pos_t>> enpassants;
  std::array<ply_index_t, 4> castlings = {board::nocastlings, board::nocastlings, board::nocastlings, board::nocastlings};
  std::vector<ply_index_t> halfmoves;

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
    board_mailbox_t pins_rays;
  };
  board_state state;
  std::vector<board_state> state_hist;

  std::array<pos_t, 2> pos_king = {piece::uninitialized_king, piece::uninitialized_king};
  ply_index_t state_hist_repetitions = ~ply_index_t(0);

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
    halfmoves({f.halfmove_clock})
  {
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
    if(!(f.castling_compressed & (1 << 0)))unset_castling(BLACK,QUEEN_SIDE);
    if(!(f.castling_compressed & (1 << 1)))unset_castling(BLACK,KING_SIDE);
    if(!(f.castling_compressed & (1 << 2)))unset_castling(WHITE,QUEEN_SIDE);
    if(!(f.castling_compressed & (1 << 3)))unset_castling(WHITE,KING_SIDE);
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
    const pos_t etrace = enpassant_trace();
    uint8_t enpassant_castlings = (etrace == event::enpassantnotrace) ? 0x0f : board::_x(etrace);
    enpassant_castlings |= get_castlings_compressed() << 4;
    state.info = (board_info){
      .active_player=activePlayer(),
      .enpassant_castlings=enpassant_castlings,
      .whites=bits[WHITE],
      .blacks=bits[BLACK],
      .diag_slid=bits_slid_diag,
      .orth_slid=bits_slid_orth,
      .pawns=bits_pawns
             | (uint64_t(pos_king[BLACK]) << 7*board::LEN) \
             | uint64_t(pos_king[WHITE]),
    };
  }

  INLINE board_info get_board_info() const {
    return state.info;
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
//        pos_king[WHITE] = piece::uninitialized_king;
//      }
    } else if(bits[BLACK] & i_mask) {
      piece::unset_pos(bits[BLACK], i);
//      if(self[i].value == KING && pos_king[BLACK] == i) {
//        pos_king[BLACK] = piece::uninitialized_king;
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
    assert(c != NEUTRAL);
    return pos_king[c] == i && piece::is_king_castling_move(c, i, j);
  }

  INLINE bool is_doublepush_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    if(~bits_pawns & piece::pos_mask(i))return false;
    const COLOR c = self.color_at_pos(i);
    return piece::is_pawn_double_push(c, i, j);
  }

  INLINE bool is_enpassant_take_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    return (bits_pawns & piece::pos_mask(i)) && j == enpassant_trace() && j != event::enpassantnotrace;
  }

  INLINE bool is_promotion_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    if(~bits_pawns & piece::pos_mask(i))return false;
    const COLOR c = self.color_at_pos(i);
    return piece::is_pawn_promotion_move(c, i, j);
  }

  INLINE bool is_capture_move(pos_t i, pos_t j) const {
    j &= board::MOVEMASK;
    return bits[enemy_of(activePlayer())] & piece::pos_mask(j);
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
//  INLINE void move_pos1(pos_t i, pos_t j) {
//    assert(j <= board::MOVEMASK);
//    put_pos(j, self[i]);
//    unset_pos(i);
//  }
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

  event_t ev_basic(pos_t i, pos_t j) const {
    const pos_t promote_flag = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    assert((bits[WHITE]|bits[BLACK]) & piece::pos_mask(i));
    pos_t killwhat = event::killnothing;
    pos_t enpassant_trace = event::enpassantnotrace;
    if(!self.empty_at_pos(j)) {
      killwhat = self[j].piece_index;
    }
    if(is_doublepush_move(i, j)) {
      const COLOR c = self.color_at_pos(i);
      enpassant_trace=piece::get_pawn_enpassant_trace(c, i, j);
    }
    return event::basic(bitmask::_pos_pair(i, j | promote_flag), killwhat, enpassant_trace);
  }

  INLINE event_t ev_castle(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    pos_pair_t rookmove = 0x00;
    if(bits[WHITE] & piece::pos_mask(i))rookmove = piece::get_king_castle_rook_move(WHITE, i, j);
    if(bits[BLACK] & piece::pos_mask(i))rookmove = piece::get_king_castle_rook_move(BLACK, i, j);
    const pos_t r_i = bitmask::first(rookmove),
                r_j = bitmask::second(rookmove);
    return event::castling(bitmask::_pos_pair(i, j), bitmask::_pos_pair(r_i, r_j));
  }

  INLINE event_t ev_take_enpassant(pos_t i, pos_t j) const {
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

  INLINE void set_enpassant(pos_t e) {
    if(e != event::enpassantnotrace) {
      enpassants.emplace_back(get_current_ply(), e);
    }
  }

  INLINE event_t ev_promotion(pos_t i, pos_t j) const {
    return event::promotion_from_basic(ev_basic(i, j));
  }

  INLINE pos_t get_halfmoves() const {
    return get_current_ply() - halfmoves.back();
  }

  INLINE void update_halfmoves() {
    halfmoves.emplace_back(get_current_ply());
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
    const COLOR c = self.color_at_pos(i);
    const pos_t castlrank = (c == WHITE) ? 1 : 8;
    const piece_bitboard_t rooks = bits_slid_orth & ~bits_slid_diag;
    if(pos_king[c] == i) {
      unset_castling(c, KING_SIDE);
      unset_castling(c, QUEEN_SIDE);
    } else if(rooks & piece::pos_mask(i)) {
      if(is_castling(c, QUEEN_SIDE) && i == board::_pos(A, castlrank)) {
        unset_castling(c, QUEEN_SIDE);
      } else if(is_castling(c, KING_SIDE) && i == board::_pos(H, castlrank)) {
        unset_castling(c, KING_SIDE);
      }
    }

    if(rooks & piece::pos_mask(j)) {
      const COLOR ec = enemy_of(c);
      const pos_t ecastlrank = (c == BLACK) ? 1 : 8;
      if(is_castling(ec, QUEEN_SIDE) && j == board::_pos(A, ecastlrank)) {
        unset_castling(ec, QUEEN_SIDE);
      } else if(is_castling(ec, KING_SIDE) && j == board::_pos(H, ecastlrank)) {
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
    if(enpassant_trace() != event::enpassantnotrace) {
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
    return i == (i & board::MOVEMASK) &&
           (bits[activePlayer()] & piece::pos_mask(i)) &&
           self.get_moves_from(i) & piece::pos_mask(j & board::MOVEMASK);
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

  void make_move(move_t m) {
    event_t ev = get_move_event(m);
    ++current_ply_;
    const uint8_t marker = event::extract_marker(ev);
    assert(fen::decompress_castlings(fen::compress_castlings(get_castlings_mask())) == get_castlings_mask());
    state_hist.emplace_back(state);
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
          if(killwhat != event::killnothing || bits_pawns & piece::pos_mask(i)) {
            update_halfmoves();
          }
          update_castlings(i, j);
          if(killwhat == event::killnothing) {
            move_pos_quiet(i, j);
          } else {
            move_pos(i, j);
          }
          update_state_attacks_pos(i);
          update_state_attacks_pos(j);
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
          move_pos_quiet(i, j);
          update_state_attacks_pos(i);
          update_state_attacks_pos(j);
          move_pos_quiet(r_i, r_j);
          update_state_attacks_pos(r_i);
          update_state_attacks_pos(r_j);
        }
      break;
      case event::ENPASSANT_MARKER:
        {
          const move_t m = event::extract_move(ev);
          const pos_t i = bitmask::first(m),
                      j = bitmask::second(m);
          const pos_t killwhere = event::extract_pos(ev);
          put_pos(killwhere, self.get_piece(EMPTY));
          update_state_attacks_pos(killwhere);
          move_pos_quiet(i, j);
          update_state_attacks_pos(i);
          update_state_attacks_pos(j);
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
          unset_pos(i);
          put_pos(j, self.pieces[Piece::get_piece_index(becomewhat, activePlayer())]);
          update_state_attacks_pos(i);
          update_state_attacks_pos(j);
          update_halfmoves();
        }
      break;
    }
    activePlayer_ = enemy_of(activePlayer());
    init_state_moves();
    update_state_info();
    if(state_hist_repetitions > self.get_current_ply()) {
      update_state_repetitions(3);
    }
  }

  INLINE void make_move(pos_t i, pos_t j) {
    assert(check_valid_move(i, j));
    make_move(bitmask::_pos_pair(i, j));
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
    while(!enpassants.empty() && enpassants.back().first > get_current_ply()) {
      enpassants.pop_back();
    }
    // castlings
    for(pos_t cstl = 0; cstl < castlings.size(); ++cstl) {
      if(castlings[cstl] > get_current_ply()) {
        castlings[cstl] = board::nocastlings;
      }
    }
    // halfmoves
    if(halfmoves.size() > 1) {
      while(halfmoves.back() > get_current_ply()) {
        halfmoves.pop_back();
      }
    }
    // state
    state = state_hist.back();
    state_hist.pop_back();
    if(self.get_current_ply() < state_hist_repetitions) {
      state_hist_repetitions = ~ply_index_t(0);
    }
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
    return is_draw() || !can_move();
  }

  void init_update_state() {
    init_state_attacks();
    init_state_moves();
    halfmoves.reserve(piece::size(bits[WHITE] | bits[BLACK]) - 2);
    enpassants.reserve(16);
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

  INLINE piece_bitboard_t get_attacks_from(pos_t pos) const { return state.attacks[pos]; }

  INLINE piece_bitboard_t get_sliding_diag_attacks_to(pos_t j, piece_bitboard_t occupied, COLOR c) const {
    assert(c == BOTH || c < NO_COLORS);
    const piece_bitboard_t colormask = (c == BOTH) ? occupied : occupied & bits[c];
    return piece::get_sliding_diag_attack(j,occupied) & bits_slid_diag & colormask;
  }

  INLINE piece_bitboard_t get_sliding_orth_attacks_to(pos_t j, piece_bitboard_t occupied, COLOR c) const {
    assert(c == BOTH || c < NO_COLORS);
    const piece_bitboard_t colormask = (c == BOTH) ? occupied : occupied & bits[c];
    return piece::get_sliding_orth_attack(j,occupied) & bits_slid_orth & colormask;
  }

  INLINE piece_bitboard_t get_sliding_attacks_to(pos_t j, piece_bitboard_t occupied, COLOR c) const {
    assert(c == BOTH || c < NO_COLORS);
    return get_sliding_diag_attacks_to(j, occupied, c) | get_sliding_orth_attacks_to(j, occupied, c);
  }

  INLINE piece_bitboard_t get_pawn_attacks_to(pos_t j, COLOR c) const {
    assert(c == BOTH || c < NO_COLORS);
    if(c==BOTH)return get_pawn_attacks_to(j, WHITE) | get_pawn_attacks_to(j, BLACK);
    return piece::get_pawn_attack(j,enemy_of(c)) & (bits_pawns & bits[c]);
  }

  INLINE piece_bitboard_t get_king_attacks_to(pos_t j, COLOR c) const {
    assert(c == BOTH || c < NO_COLORS);
    const piece_bitboard_t kingmask = (c == BOTH) ? get_king_bits() : piece::pos_mask(pos_king[c]);
    return piece::get_king_attack(j) & kingmask;
  }

  INLINE piece_bitboard_t get_attacks_to(pos_t j, COLOR c, piece_bitboard_t occ_mask=~0ULL) const {
    assert(c == BOTH || c < NO_COLORS);
    const piece_bitboard_t occupied = (bits[WHITE] | bits[BLACK]) & occ_mask;
    const piece_bitboard_t colormask = (c == BOTH) ? occupied : bits[c];
    return get_sliding_attacks_to(j, occupied, c)
      | (piece::get_knight_attack(j) & get_knight_bits() & colormask)
      | get_king_attacks_to(j, c)
      | get_pawn_attacks_to(j, c);
  }

  INLINE piece_bitboard_t get_attack_counts_to(pos_t j, COLOR c) const {
    return bitmask::count_bits(get_attacks_to(j,c));
  }

  void update_state_attacks_pos(pos_t pos) {
    const piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    const piece_bitboard_t affected = piece::pos_mask(pos) | get_sliding_attacks_to(pos, occupied, BOTH);

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
    const piece_bitboard_t attackers = get_attacks_to(pos_king[c], enemy_of(c));
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
    if(state.checkline[c] == ~0ULL || (get_attacks_from(pos_king[c]) & ~bits[c] & ~get_attack_mask(enemy_of(c)))) {
      return false;
    }
    return !is_draw() && !can_move();
  }

  INLINE bool can_move() const {
    const COLOR c = activePlayer();
    bool canmove = false;
    bitmask::foreach_early_stop(bits[c], [&](pos_t pos) mutable -> bool {
      if(get_moves_from(pos))canmove=true;
      return !canmove;
    });
    return canmove;
  }

  INLINE bool is_draw_stalemate() const {
    const COLOR c = activePlayer();
    return !get_attacks_to(pos_king[c], enemy_of(c)) && !can_move();
  }

  INLINE bool is_draw_material() const {
    if(bits_slid_orth)return false;
    const size_t no_pieces = piece::size(bits[WHITE] | bits[BLACK]);
    const piece_bitboard_t bishops = bits_slid_diag,
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

  INLINE void update_state_repetitions(int no_times=3) {
    if(no_times >= 3 && state_hist_repetitions <= self.get_current_ply()) {
      state_hist_repetitions = self.get_current_ply();
      return;
    }
    int repetitions = 1;
    for(ply_index_t i = 0; i < get_halfmoves(); ++i) {
      if(state.info == state_hist[state_hist.size() - i - 1].info) {
        ++repetitions;
        if(repetitions >= no_times) {
          state_hist_repetitions = self.get_current_ply();
          return;
        }
      }
    }
    return;
  }

  INLINE bool can_draw_repetition() const {
    return state_hist_repetitions <= self.get_current_ply();
  }

  ALWAYS_UNROLL void init_state_moves() {
    for(auto&m:state.moves)m=0x00;
    if(is_draw_halfmoves()||is_draw_material())return;
    const COLOR c = activePlayer();

    {
      init_state_checkline(c);
      const COLOR ec = enemy_of(c);
      const piece_bitboard_t friends = bits[c], foes = bits[ec],
                             attack_mask = get_attack_mask(ec);
      const bool doublecheck = (state.checkline[c] == 0x00);
      if(!doublecheck) {
        bitmask::foreach(bits_pawns & bits[c], [&](pos_t pos) mutable noexcept -> void {
          piece_bitboard_t foes_pawn = foes;
          if(enpassant_trace() != event::enpassantnotrace) {
            foes_pawn |= piece::pos_mask(enpassant_trace());
          }
          state.moves[pos] = get_attacks_from(pos) & foes_pawn;
          state.moves[pos] |= piece::get_pawn_push_moves(c, pos, friends|foes);
          state.moves[pos] &= state.checkline[c];
        });
        bitmask::foreach(bits[c] & ~(bits_pawns | get_king_bits()), [&](pos_t pos) mutable noexcept -> void {
          state.moves[pos] = get_attacks_from(pos) & ~friends;
          state.moves[pos] &= state.checkline[c];
        });
      }
      state.moves[pos_king[c]] = get_attacks_from(pos_king[c]) & ~friends & ~attack_mask;
      state.moves[pos_king[c]] |= piece::get_king_castling_moves(c, pos_king[c], friends|foes, attack_mask, get_castlings_mask());
      init_state_moves_checkline_enpassant_takes();
      init_state_pins();
    }
  }

  void init_state_moves_checkline_enpassant_takes() {
    const COLOR c = activePlayer();
    const pos_t etrace = enpassant_trace();
    const pos_t epawn = enpassant_pawn();
    if(etrace == event::enpassantnotrace)return;
    if(!bitmask::is_exp2(state.checkline[c]))return;
    const pos_t attacker = bitmask::log2_of_exp2(state.checkline[c]);
    if(epawn != attacker)return;
    const piece_bitboard_t apawns = get_pawn_attacks_to(etrace,c);
    if(!apawns)return;
    bitmask::foreach(apawns, [&](pos_t apawn) mutable noexcept -> void {
      state.moves[apawn] |= piece::pos_mask(etrace);
    });
  }

  INLINE piece_bitboard_t get_pins(COLOR c) const {
    assert(c < NO_COLORS);
    return state.pins[c];
  }

  template <typename F>
  INLINE void iter_attacking_xrays(pos_t j, F &&func, COLOR c) const {
    assert(c < NO_COLORS);
    const piece_bitboard_t dstbit = piece::pos_mask(j);
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t friends = bits[c],
                           foes = bits[ec];
    const piece_bitboard_t rook_xray = piece::get_sliding_orth_xray_attack(j, foes, friends) & bits[ec] & bits_slid_orth;
    if(rook_xray) {
      bitmask::foreach(rook_xray, [&](pos_t i) mutable -> void {
        if(bits_slid_orth & piece::pos_mask(i)) {
          const piece_bitboard_t srcbit = piece::pos_mask(j);
          const piece_bitboard_t r = piece::get_sliding_orth_attacking_xray(i,j,foes|dstbit,friends) & ~srcbit;
          func(i, r | dstbit);
        }
      });
    }
    const piece_bitboard_t bishop_xray = piece::get_sliding_diag_xray_attack(j, foes, friends) & bits[ec] & bits_slid_diag;
    if(bishop_xray) {
      bitmask::foreach(bishop_xray, [&](pos_t i) mutable -> void {
        if(bits_slid_diag & piece::pos_mask(i)) {
          const piece_bitboard_t srcbit = piece::pos_mask(j);
          const piece_bitboard_t r = piece::get_sliding_diag_attacking_xray(i,j,foes|dstbit,friends) & ~srcbit;
          func(i, r | dstbit);
        }
      });
    }
  }

  void init_state_pins() {
    for(auto &spr : state.pins_rays)spr=~0x00ULL;
    const COLOR c = activePlayer();
    // maybe do this as a loop for incremental updates
    {
      state.pins[c] = 0x00;
      const piece_bitboard_t kingmask = piece::pos_mask(pos_king[c]);
      const piece_bitboard_t friends = bits[c] & ~kingmask;
      iter_attacking_xrays(pos_king[c], [&](pos_t i, piece_bitboard_t r) mutable -> void {
        const pos_t attacker = i;
        r |= piece::pos_mask(attacker);
        r &= ~kingmask;
        const piece_bitboard_t pin = friends & r;
        if(pin) {
          assert(bitmask::is_exp2(pin));
          const pos_t pin_pos = bitmask::log2_of_exp2(pin);
          state.pins[c] |= pin;
          state.pins_rays[pin_pos] = r;
          // update moves
          state.moves[pin_pos] &= r;
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
      const COLOR ec = enemy_of(c);
      if(state.checkline[c] != ~0ULL)return;
      const pos_t etrace = enpassant_trace();
      const pos_t epawn = enpassant_pawn();
      const piece_bitboard_t h = piece::rank_mask(board::_y(pos_king[c]));
      if(!(h & bits_slid_orth & bits[ec]))return;
      if(!(h & piece::pos_mask(epawn)))return;
      const piece_bitboard_t apawns = get_pawn_attacks_to(etrace,c);
      if(!apawns)return;
      bitmask::foreach(apawns, [&](pos_t apawn) mutable -> void {
        piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
        occupied &= ~(piece::pos_mask(apawn) | piece::pos_mask(epawn));
        //printf("consider horizontal pin %hhu -> %hhu\n", apawn, etrace);
        if(get_sliding_orth_attacks_to(pos_king[c],occupied,ec)) {
          //printf("horizontal pin disable %hhu -> %hhu\n", apawn, etrace);
          state.pins[self.color_at_pos(apawn)] |= piece::pos_mask(apawn);
          const piece_bitboard_t forbid = ~piece::pos_mask(etrace);
          state.pins_rays[apawn] &= forbid;
          state.moves[apawn] &= forbid;
        }
      });
    }
  }

  INLINE piece_bitboard_t get_moves_from(pos_t pos) const { return state.moves[pos]; }

  void print() const {
    for(pos_t i = board::LEN; i > 0; --i) {
      for(pos_t j = 0; j < board::LEN; ++j) {
        const Piece p = self[(i-1) * board::LEN + j];
        std::cout << p.str() << " ";
      }
      std::cout << std::endl;
    }
  }

  std::string _move_str(move_t m) const {
    if(m==board::nomove)return "0000"s;
    return board::_move_str(m, bits_pawns & piece::pos_mask(bitmask::first(m)));
  }

  std::vector<std::string> _line_str(const MoveLine &line, bool thorough=false) {
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

  std::string _line_str_full(const MoveLine &line, bool thorough_fut=false) {
    std::string s = "["s + str::join(_line_str(line.get_past(), false), " "s) + "]"s;
    if(!line.empty()) {
      s += " "s + str::join(_line_str(line.get_future(), thorough_fut), " "s);
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
      .fullmove = uint16_t((get_current_ply() / 2) + 1),
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
