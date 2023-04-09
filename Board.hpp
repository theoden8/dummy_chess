#pragma once


#ifdef FLAG_JEMALLOC_EXTERNAL
#include <jemalloc/jemalloc.h>
#endif

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
  // general chess960-specific variables
  const bool chess960 = false;
  std::array<pos_t, NO_COLORS>
    kcastlrook = {0xff, 0xff},
    qcastlrook = {0xff, 0xff};

  // general crazyhouse-specific variables
  const bool crazyhouse = false;

  // cheap abstractions, pieces[Piece::piece_index(PAWN, BLACK)]
  static constexpr std::array<Piece, board::NO_PIECE_INDICES+1>  pieces = {
    Piece(PAWN, WHITE), Piece(PAWN, BLACK),
    Piece(KNIGHT, WHITE), Piece(KNIGHT, BLACK),
    Piece(BISHOP, WHITE), Piece(BISHOP, BLACK),
    Piece(ROOK, WHITE), Piece(ROOK, BLACK),
    Piece(QUEEN, WHITE), Piece(QUEEN, BLACK),
    Piece(KING, WHITE), Piece(KING, BLACK),
    Piece(EMPTY, NEUTRAL)
  };

  // compressed (partially incomplete) board state for hashing
  struct board_info {
    COLOR active_player = NEUTRAL;
    pos_t enpassant;
    pos_pair_t castlings;
    piece_bitboard_t whites, blacks, diag_slid, orth_slid, pawns;
    uint64_t n_subs_mask;
    piece_bitboard_t promoted_pawns;

    INLINE bool operator==(board_info other) const noexcept {
      return active_player == other.active_player &&
             pawns == other.pawns &&
             whites == other.whites && blacks == other.blacks &&
             diag_slid == other.diag_slid && orth_slid == other.orth_slid &&
             castlings == other.castlings && enpassant == other.enpassant &&
             n_subs_mask == other.n_subs_mask && promoted_pawns == other.promoted_pawns;
    }

    // efficiently invalidate state for comparison purposes
    INLINE void unset() {
      active_player = NEUTRAL;
    }

    // check whether state is valid
    INLINE bool is_unset() const {
      return active_player == NEUTRAL;
    }

    // uncompressed king position
    INLINE pos_t pos_king(COLOR c) const {
      assert(c < NO_COLORS);
      if(c == WHITE) {
        return pos_t(pawns);
      } else {
        return pos_t(pawns >> (7 * board::LEN));
      }
    }

    // uncompressed pawn positions
    INLINE piece_bitboard_t pawn_bits() const {
      return pawns & ((bitmask::full >> (board::LEN * 2)) << board::LEN);
    }

    // 64-bit word, each 8 bit word specifies a number of substitution pieces for the type
    std::array<pos_t, board::NO_DROPPIECE_INDICES> get_n_subs() const {
      std::array<pos_t, board::NO_DROPPIECE_INDICES> nsubs;
      for(pos_t i = 0; i < board::NO_DROPPIECE_INDICES; ++i) {
        //nsubs[i] = 0;
        nsubs[i] = (n_subs_mask >> (i * 6)) & (board::SIZE - 1);
      }
      return nsubs;
    }

    // get_n_subs externally interfaced
    static std::array<pos_t, board::NO_DROPPIECE_INDICES> get_n_subs(piece_bitboard_t n_subs_mask) {
      std::array<pos_t, board::NO_DROPPIECE_INDICES> nsubs;
      for(pos_t i = 0; i < board::NO_DROPPIECE_INDICES; ++i) {
        //nsubs[i] = 0;
        nsubs[i] = (n_subs_mask >> (i * 6)) & (board::SIZE - 1);
      }
      return nsubs;
    }
  };
public:
  // irregular state variables:
  // plies at which en-passants are set
  std::vector<std::pair<ply_index_t, pos_t>> state_hist_enpassants;
  // KQkq
  std::array<ply_index_t, 4> castlings = {board::nocastlings, board::nocastlings, board::nocastlings, board::nocastlings};
  // at which moves halfmove clock is reset
  std::vector<ply_index_t> state_hist_halfmoves;
  // when three-fold repetition occured:
  ply_index_t state_hist_repetitions = INT16_MAX;

  // bitboard representation
  std::array<piece_bitboard_t, NO_COLORS> bits = {0x00, 0x00};
  piece_bitboard_t bits_slid_diag = 0x00, bits_slid_orth = 0x00, bits_pawns = 0x00;
  std::array<pos_t, NO_COLORS> pos_king = {board::nopos, board::nopos};

  // crazyhouse-specific state variables
  piece_bitboard_t bits_promoted_pawns = 0ULL;
  std::array<pos_t, board::NO_DROPPIECE_INDICES> n_subs = {0};

  // regular board state variables
  struct board_state {
    // mailbox can represent any bipartite board graph
    using board_mailbox_t = std::array<piece_bitboard_t, board::SIZE>;
    // null moves are fake, and should not trigger repetition and such
    bool null_move_state;
    // compressed board state: bijectively restore it when needed
    board_info info;
    board_mailbox_t attacks;
    bool moves_initialized = false;
    std::array<piece_bitboard_t, NO_COLORS> checkline = {0x00,0x00};
    board_mailbox_t moves;
    std::array<piece_bitboard_t, NO_COLORS> pins;
  };
  // current state
  board_state state;
  // (only) previous states, for fast unmaking
  std::vector<board_state> state_hist;

  size_t zobrist_size;
  bool b_finalize = false;
  explicit Board(const fen::FEN &f=fen::starting_pos, size_t zbsize=ZOBRIST_SIZE):
    activePlayer_(f.active_player),
    current_ply_(f.fullmove * 2 - (f.active_player == WHITE ? 1 : 0)),
    chess960(f.chess960), crazyhouse(f.crazyhouse),
    zobrist_size(bitmask::highest_bit(zbsize))
  {
    // external initializations (singleton)
    if(!initialized_) {
      M42::init();
      zobrist::init(zobrist_size);
      initialized_ = true;
    }
    // set bitboards
    for(pos_t i = 0; i < f.board.length(); ++i) {
      if(f.board[i]==' ')continue;
      assert(i < board::SIZE);
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
      const pos_t x = board::_x(i), y = board::LEN - board::_y(i) - 1;
      const pos_t pos = board::_pos(A+x, 1+y);
      put_pos(pos, Piece(p, c));
      if(f.crazyhouse && piece::is_set(f.crazyhouse_promoted, i)) {
        piece::set_pos(self.bits_promoted_pawns, pos);
      }
    }
    // crazyhouse: set substitution counts
    if(crazyhouse) {
      for(COLOR c : {WHITE, BLACK}) {
        for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
          const Piece piece = Piece(p, c);
          n_subs[piece.piece_index] = std::count(f.subs.begin(), f.subs.end(), piece.str());
        }
      }
    }
    // set white castlings state (variation and position)
    if(bitmask::first(f.castlings)) {
      const pos_t wcastlmask = bitmask::first(f.castlings);
      if(bitmask::log2_lsb(wcastlmask) < board::_x(pos_king[WHITE])) {
        qcastlrook[WHITE] = bitmask::log2_lsb(wcastlmask);
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
    // set black castlings state (variation and position)
    if(bitmask::second(f.castlings)) {
      const pos_t bcastlmask = bitmask::second(f.castlings);
      if(bitmask::log2_lsb(bcastlmask) < board::_x(pos_king[BLACK])) {
        qcastlrook[BLACK] = bitmask::log2_lsb(bcastlmask);
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
    // initialize irregular state variables
    state_hist_halfmoves.emplace_back(current_ply_ - f.halfmove_clock);
    set_enpassant(f.enpassant);
    state_hist_halfmoves.reserve(piece::size(bits[WHITE] | bits[BLACK]) - 2);
    state_hist_enpassants.reserve(16);
    // initialize regular state variables
    state_hist.reserve(100);
    init_state_attacks();
    clear_state_unfinalize();
    init_state_checkline(activePlayer());
    init_state_moves();
    update_state_info();
    state.null_move_state = false;
  }

  INLINE constexpr COLOR activePlayer() const {
    return activePlayer_;
  }

  // return piece-type interface, expensive
  INLINE const Piece operator[](pos_t ind) const {
    assert(ind <= board::MOVEMASK);
    const piece_bitboard_t ind_mask = piece::pos_mask(ind);
    const COLOR c = self.color_at_pos(ind);
    if(c == NEUTRAL) {
      return Piece(EMPTY, NEUTRAL);
    }
    if(bits_pawns & ind_mask) {
      return Piece(PAWN, c);
    } else if(bits_slid_diag & bits_slid_orth & ind_mask) {
      return Piece(QUEEN, c);
    } else if(bits_slid_diag & ind_mask) {
      return Piece(BISHOP, c);
    } else if(bits_slid_orth & ind_mask) {
      return Piece(ROOK, c);
    } else if(pos_king[c] == ind) {
      return Piece(KING, c);
    } else {
      return Piece(KNIGHT, c);
    }
    abort();
  }

  // cheap lookup for emptiness
  INLINE bool empty_at_pos(pos_t ind) const {
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

  // update compressed (incomplete) state information
  INLINE void update_state_info() {
    uint64_t n_subs_mask = 0x00;
    if(crazyhouse) {
      for(pos_t i = 0; i < board::NO_DROPPIECE_INDICES; ++i) {
        n_subs_mask |= uint64_t(n_subs[i]) << (i * 6);
      }
      assert(n_subs == board_info::get_n_subs(n_subs_mask));
    }
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
      .n_subs_mask=n_subs_mask,
      .promoted_pawns=bits_promoted_pawns
    };
  }

  // methods to read and update bitboards
  // obtain kings bitboard
  INLINE piece_bitboard_t get_king_bits() const {
    return piece::pos_mask(pos_king[WHITE]) | piece::pos_mask(pos_king[BLACK]);
  }

  // infer knights bitboard
  INLINE piece_bitboard_t get_knight_bits() const {
    return (bits[WHITE] | bits[BLACK]) ^ (bits_slid_diag | bits_slid_orth | bits_pawns | get_king_bits());
  }

  // remove i from bitboard representation
  INLINE void unset_pos(pos_t i) {
    assert(i < 64);
    const piece_bitboard_t i_mask = piece::pos_mask(i);
    if(bits[WHITE] & i_mask) {
      piece::unset_pos(bits[WHITE], i);
//      if(self[i].value == KING && pos_king[WHITE] == i) {
//        pos_king[WHITE] = board::nopos;
//      }
    } else if(bits[BLACK] & i_mask) {
      piece::unset_pos(bits[BLACK], i);
//      if(self[i].value == KING && pos_king[BLACK] == i) {
//        pos_king[BLACK] = board::nopos;
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

  // add i to bitboard representation (for the given piece type and color)
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

  // replace whatever is at i with a given piece-type
  INLINE void put_pos(pos_t i, const Piece p) {
    unset_pos(i);
    set_pos(i, p);
  }

  // move i to j, emptiness at j guaranteed
  INLINE void move_pos_quiet(pos_t i, pos_t j) {
    assert(self.empty_at_pos(j));
    // can't be promotion
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

  // methods to determine move types
  INLINE bool is_castling_move(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
    const COLOR c = self.color_at_pos(i);
    const pos_t castlrank = (c == WHITE) ? 1 : 8;
    if(!chess960) {
      return (i == pos_king[c]) && (i == board::_pos(E, castlrank))
          && (j == board::_pos(C, castlrank) || j == board::_pos(G, castlrank));
    } else {
      return (i == pos_king[c]) && (bits_slid_orth & ~bits_slid_diag & bits[c] & piece::pos_mask(j));
    }
  }

  INLINE bool is_doublepush_move(pos_t i, pos_t j) const {
    assert(i <= board::MOVEMASK && j <= board::MOVEMASK);
    if(~bits_pawns & piece::pos_mask(i))return false;
    const COLOR c = self.color_at_pos(i);
    return piece::is_pawn_double_push(c, i, j);
  }

  INLINE bool is_drop_move(pos_t i, pos_t j) const {
    return i & board::CRAZYHOUSE_DROP;
  }

  INLINE bool is_enpassant_take_move(pos_t i, pos_t j) const {
    assert(i <= board::MOVEMASK && j <= board::MOVEMASK);
    return (bits_pawns & piece::pos_mask(i)) && j == enpassant_trace();
  }

  INLINE bool is_promotion_move(pos_t i, pos_t j) const {
    assert(i <= board::MOVEMASK && j <= board::MOVEMASK);
    if(~bits_pawns & piece::pos_mask(i))return false;
    const COLOR c = self.color_at_pos(i);
    return piece::is_pawn_promotion_move(c, i, j);
  }

  INLINE bool is_naively_capture_move(pos_t i, pos_t j) const {
    assert(i <= board::MOVEMASK && j <= board::MOVEMASK);
    const COLOR c = self.color_at_pos(i);
    return bits[enemy_of(c)] & piece::pos_mask(j);
  }

  INLINE bool is_naively_capture_move(move_t m) const {
    return is_naively_capture_move(bitmask::first(m), bitmask::second(m));
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

  // methods to read and update irregular state variables
  INLINE pos_t enpassant_trace() const {
    if(state_hist_enpassants.empty())return board::nopos;
    const auto [ply, e] = state_hist_enpassants.back();
    if(ply == get_current_ply()) {
      return e;
    }
    return board::nopos;
  }

  INLINE pos_t enpassant_pawn() const {
    if(enpassant_trace() == board::nopos)return 0xFF;
    const pos_t x = board::_x(enpassant_trace());
    return board::_y(enpassant_trace()) == 3-1 ? board::_pos(A+x, 4) : board::_pos(A+x, 5);
  }

  INLINE void set_enpassant(pos_t e) {
    if(e != board::nopos) {
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

  // methods to get zobrist hashing
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

  INLINE zobrist::key_t zb_hash() const {
    return zobrist::zb_hash(self);
  }

  INLINE zobrist::key_t zb_hash_material(bool mirror=false) const {
    return zobrist::zb_hash_material(self, mirror);
  }

  INLINE ply_index_t get_current_ply() const {
    return current_ply_;
  }

  // methods to check move validity
  INLINE bool check_valid_drop_move(pos_t i, pos_t j) const {
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    PIECE p = board::get_drop_as(i);
    constexpr piece_bitboard_t non_final_ranks = piece::rank_mask(1) | piece::rank_mask(2) | piece::rank_mask(3)
      | piece::rank_mask(4) | piece::rank_mask(5) | piece::rank_mask(6);
    const piece_bitboard_t drop_locations = ~(bits[c] | bits[ec]) & state.checkline[c];
    return crazyhouse
          && is_drop_move(i, j)
          && pos_t(p) < pos_t(NO_PIECES)
          && j <= board::MOVEMASK
          && n_subs[Piece::get_piece_index(board::get_drop_as(i), c)] > 0
          && (
            (p != PAWN && (drop_locations & piece::pos_mask(j)))
            || (drop_locations & non_final_ranks & piece::pos_mask(j))
          );
  }

  INLINE bool check_valid_move(pos_t i, pos_t j, bool strict=true) {
#ifndef NDEBUG
    const bool tmp = !state.moves_initialized;
    if(tmp) {
      return true;
    }
#endif
    return (crazyhouse && check_valid_drop_move(i, j))
      || (!strict && bitmask::_pos_pair(i, j) == board::nullmove)
      || (i == (i & board::MOVEMASK)
           && (bits[activePlayer()] & piece::pos_mask(i))
           && (state.moves[i] & piece::pos_mask(j & board::MOVEMASK)));
  }

  INLINE bool check_valid_move(const move_t m, bool strict=true) {
    return check_valid_move(bitmask::first(m), bitmask::second(m), strict);
  }

  // visitor methods for the engine
  virtual void _restore_on_event() {}
  virtual void _update_pos_change(pos_t i, pos_t j) {};

  void make_move_unfinalized(pos_t i, pos_t j) {
    const move_t m = bitmask::_pos_pair(i, j);
    // split promotion and destination information
    const pos_t promote_as = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    // some move properties, which depend on increasing ply counter
    const bool isdrop = crazyhouse && is_drop_move(i, j);
    const bool known_move_type = (m == board::nullmove || isdrop);
    const bool iscastling = !known_move_type && is_castling_move(i, j);
    const bool is_enpassant_take =  !known_move_type && is_enpassant_take_move(i, j);
    const pos_t epawn = enpassant_pawn();

    // back-up current state
    state_hist.emplace_back(state);
    assert(check_valid_move(m, false));
    state.null_move_state = (m == board::nullmove);
    ++current_ply_;
    if(m == board::nullmove) {
      // do nothing
    } else if(isdrop) {
      // crazyhouse-specific move
      assert(check_valid_drop_move(i, j));
      const COLOR c = activePlayer();
      const PIECE p = board::get_drop_as(i);
      put_pos(j, Piece(p, c));
      update_state_attacks_pos(j);
      --n_subs[Piece::get_piece_index(p, c)];
      state_hist_halfmoves.emplace_back(get_current_ply());
    } else if(iscastling) {
      // back up knights for later checksum, as none of the knights should move
      const piece_bitboard_t knights = get_knight_bits();
      const COLOR c = activePlayer();
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      // king moves k_i -> k_j
      const pos_t k_i = i;
      pos_t k_j = j;
      // in chess 960, j is not where the king moves, but the rook it castles towards
      // otherwise there can be ambiguity of whether the king wants to move or castle
      if(chess960) {
        if(j == board::_pos(qcastlrook[c], castlrank)) {
          k_j = board::_pos(C, castlrank);
        } else if(j == board::_pos(kcastlrook[c], castlrank)) {
          k_j = board::_pos(G, castlrank);
        } else {
          abort();
        }
      }
      // rook moves r_i -> r_j
      const move_t rookmove = piece::get_king_castle_rook_move(c, i, k_j, qcastlrook[c], kcastlrook[c]);
      const pos_t r_i = bitmask::first(rookmove),
                  r_j = bitmask::second(rookmove);
      update_castlings(i, j);
      {
        // will k_i -> k_j overwrite rook?
        // we update white and black bitboards, not piece-bitboards here
        if(k_j != r_i) {
          // what if k_i == k_j? this is why move is not necessarily quiet
          piece::move_pos(bits[c], k_i, k_j);
        } else {
          piece::unset_pos(bits[c], k_i);
        }
        // if r_i == k_j, we've moved the king but have not moved the rook yet!
        // at this moment, the rook bitboard also indicates k_j
        pos_king[c] = k_j;
      }
      // this should work nonetheless: it takes a position at k_i,
      // marks queen+knight reachable squares and updates their attacks
      update_state_attacks_pos(k_i);
      update_state_attacks_pos(k_j);
      // notify a visitor of the change
      _update_pos_change(i, k_j);
      {
        // again: is this some weird chess960 set-up where k_j == r_i?
        // first, update color masks
        if(k_j != r_i) {
          // r_j might be the same as r_i, therefore this move is not necessarily quiet
          piece::move_pos(bits[c], r_i, r_j);
        } else {
          piece::set_pos(bits[c], r_j);
        }
        // now, update the orthogonal pieces (Q|R) bitboard
        piece::move_pos(bits_slid_orth, r_i, r_j);
      }
      // this should work fine: at this point the bitboards are valid
      update_state_attacks_pos(r_i);
      update_state_attacks_pos(r_j);
      // notify a visitor of the change
      _update_pos_change(r_i, r_j);
      const COLOR ec = enemy_of(c);
      // checksum between color-bitboards and piece-bitboards
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
      if(crazyhouse) {
        ++n_subs[Piece::get_piece_index(PAWN, c)];
      }
      state_hist_halfmoves.emplace_back(get_current_ply());
    } else if(is_promotion_move(i, j)) {
      const PIECE becomewhat = board::get_promotion_as(promote_as);
      if(crazyhouse) {
        // this is crazyhouse-specific: when promoted pawns are captured,
        // in drop-value they are still pawns. bits_promoted_pawns keeps track
        // of them for both sides
        const PIECE p = self[j].value;
        if(p != EMPTY) {
          ++n_subs[Piece::get_piece_index(p, activePlayer())];
        }
        bits_promoted_pawns |= piece::pos_mask(j);
      }
      update_castlings(i, j);
      unset_pos(i);
      put_pos(j, Piece(becomewhat, activePlayer()));
      update_state_attacks_pos(i);
      update_state_attacks_pos(j);
      _update_pos_change(i, j);
      state_hist_halfmoves.emplace_back(get_current_ply());
    } else {
      const COLOR c = activePlayer();
      const bool is_capture = !self.empty_at_pos(j);
      if(is_doublepush_move(i, j)) {
        set_enpassant(piece::get_pawn_enpassant_trace(c, i, j));
      }
      if(is_capture || bits_pawns & piece::pos_mask(i)) {
        if(crazyhouse && is_capture) {
          const PIECE p = (bits_promoted_pawns & piece::pos_mask(j)) ? PAWN : self[j].value;
          ++n_subs[Piece::get_piece_index(p, c)];
        }
        state_hist_halfmoves.emplace_back(get_current_ply());
      }
      if(crazyhouse) {
        piece::unset_pos(bits_promoted_pawns, j);
        if(bits_promoted_pawns & piece::pos_mask(i)) {
          piece::move_pos(bits_promoted_pawns, i, j);
        }
      }
      update_castlings(i, j);
      if(!is_capture) {
        move_pos_quiet(i, j);
      } else {
        move_pos(i, j);
      }
      update_state_attacks_pos(i);
      update_state_attacks_pos(j);
      _update_pos_change(i, j);
    }
    // update active player (current ply was updated earler)
    activePlayer_ = enemy_of(activePlayer());
    // init_state_moves
    update_state_info();
    // repetition detection can now be updated too
    if(state_hist_repetitions > self.get_current_ply()) {
      update_state_repetitions();
    }
    clear_state_unfinalize();
    init_state_checkline(activePlayer());
    if(b_finalize)make_move_finalize();
  }

  INLINE void make_move_unfinalized(move_t m) {
    make_move_unfinalized(bitmask::first(m), bitmask::second(m));
  }

  INLINE void make_move_finalize() {
    if(!state.moves_initialized) {
      init_state_moves();
    }
  }

  // forward-pass, complete code
  INLINE void make_move(pos_t i, pos_t j) {
    make_move_unfinalized(i, j);
    // post-processing: moves, current state info
    make_move_finalize();
    // notify the visitor that they can post-process now
  }

  INLINE void make_move(move_t m) {
    make_move(bitmask::first(m), bitmask::second(m));
  }

  // backtrack, complete method
  void retract_move() {
    // this is more efficient than make_move-like code
    // just fetch previous state and overwrite
    if(state_hist.empty())return;
    const board_info prev_info = state_hist.back().info;
    bits = {prev_info.whites, prev_info.blacks};
    bits_pawns = prev_info.pawn_bits();
    bits_slid_diag = prev_info.diag_slid;
    bits_slid_orth = prev_info.orth_slid;
    pos_king = {prev_info.pos_king(WHITE), prev_info.pos_king(BLACK)};
    activePlayer_ = enemy_of(activePlayer());
    // restore substitutions and promoted pawns too in crazyhouse
    if(crazyhouse) {
      n_subs = prev_info.get_n_subs();
      bits_promoted_pawns = prev_info.promoted_pawns;
    }
    --current_ply_;
    // enpassants
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
    // regular state variables
    state = state_hist.back();
    state_hist.pop_back();
    // repetitions
    if(self.get_current_ply() < state_hist_repetitions) {
      state_hist_repetitions = INT16_MAX;
    }
    _restore_on_event();
  }

  // methods that produce scopes for making moves and series of moves
  INLINE decltype(auto) move_scope(move_t m) {
    return make_move_scope(self, m);
  }

  INLINE decltype(auto) move_unfinalized_scope(move_t m) {
    return make_move_unfinalized_scope(self, m);
  }

  INLINE decltype(auto) mline_scope(move_t m, MoveLine &mline) {
    return make_mline_scope(self, m, mline);
  }

  INLINE decltype(auto) mline_unfinalized_scope(move_t m, MoveLine &mline) {
    return make_mline_unfinalized_scope(self, m, mline);
  }

  INLINE decltype(auto) recursive_move_scope() {
    return make_recursive_move_scope(self);
  }

  INLINE decltype(auto) recursive_move_unfinalized_scope() {
    return make_recursive_move_unfinalized_scope(self);
  }

  INLINE decltype(auto) recursive_mline_scope(MoveLine &mline) {
    return make_recursive_mline_scope(self, mline);
  }

  template <typename LineT, typename F>
  INLINE void foreach_early_stop(const LineT &mline, F &&func) {
    for(const move_t m : mline) {
      if(!func(m))break;
    }
  }

  template <typename LineT, typename F>
  INLINE void walk_unfinalized_early_stop(const LineT &mline, F &&func) {
    walk_unfinalized_early_stop(mline, std::forward<F>(func), [](ply_index_t){});
  }

  template <typename LineT, typename F, typename FF>
  INLINE void walk_unfinalized_early_stop(const LineT &mline, F &&func, FF &&endfunc) {
    auto &&rec_mscope = self.recursive_move_unfinalized_scope();
    foreach_early_stop(mline, [&](const move_t m) mutable -> bool {
      return func(m, [&]() mutable -> void {
        rec_mscope.scope(m);
      });
    });
    endfunc((ply_index_t)rec_mscope.counter);
  }

  template <typename LineT, typename F>
  INLINE void walk_early_stop(const LineT &mline, F &&func) {
    walk_early_stop(mline, std::forward<F>(func), [](ply_index_t){});
  }


  template <typename LineT, typename F, typename FF>
  INLINE void walk_early_stop(const LineT &mline, F &&func, FF &&endfunc) {
    auto &&rec_mscope = self.recursive_move_scope();
    foreach_early_stop(mline, [&](const move_t m) mutable -> bool {
      return func(m, [&]() mutable -> void {
        rec_mscope.scope(m);
      });
    });
    endfunc((ply_index_t)rec_mscope.counter);
  }

  template <typename LineT, typename F>
  INLINE void foreach(const LineT &mline, F &&func) {
    for(const move_t m : mline) {
      func(m);
    }
  }

  template <typename LineT, typename F>
  INLINE void walk_unfinalized(const LineT &mline, F &&func) {
    walk_unfinalized(mline, std::forward<F>(func), [](ply_index_t){});
  }

  template <typename LineT, typename F, typename FF>
  INLINE void walk_unfinalized(const LineT &mline, F &&func, FF &&endfunc) {
    auto &&rec_mscope = self.recursive_move_unfinalized_scope();
    foreach(mline, [&](const move_t m) mutable -> void {
      func(m, [&]() mutable -> void {
        rec_mscope.scope(m);
      });
    });
    endfunc((ply_index_t)rec_mscope.counter);
  }

  template <typename LineT, typename F>
  INLINE void walk(const LineT &mline, F &&func) {
    walk(mline, std::forward<F>(func), [](ply_index_t){});
  }

  template <typename LineT, typename F, typename FF>
  INLINE void walk(const LineT &mline, F &&func, FF &&endfunc) {
    auto &&rec_mscope = self.recursive_move_scope();
    foreach(mline, [&](const move_t m) mutable -> void {
      func(m, [&]() mutable -> void {
        rec_mscope.scope(m);
      });
    });
    endfunc((ply_index_t)rec_mscope.counter);
  }

  template <typename LineT, typename FF>
  INLINE void walk_unfinalized_end(const LineT &mline, FF &&endfunc) {
    auto &&rec_mscope = self.recursive_move_scope();
    foreach(mline, [&](const move_t m) mutable -> void {
      rec_mscope.scope(m);
    });
    endfunc((ply_index_t)rec_mscope.counter);
  }

  template <typename LineT, typename FF>
  INLINE void walk_end(const LineT &mline, FF &&endfunc) {
    auto &&rec_mscope = self.recursive_move_unfinalized_scope();
    foreach(mline, [&](const move_t m) mutable -> void {
      rec_mscope.scope(m);
    });
    endfunc((ply_index_t)rec_mscope.counter);
  }

  template <typename LineT>
  INLINE bool check_valid_sequence(const LineT &mline, bool strict=false) {
    //make_move_finalize();
    auto &&rec_mscope = recursive_move_scope();
    for(const move_t m : mline) {
      if(!state.moves_initialized) {
        make_move_finalize();
        const bool res = check_valid_move(m, false);
        clear_state_unfinalize();
        if(!res) {
          return false;
        }
      } else if(!check_valid_move(m, false)) {
        return false;
      }
      rec_mscope.scope(m);
    }
    return true;
  }

  template <typename LineT>
  INLINE bool check_line_terminates(const LineT &mline) {
    make_move_finalize();
    auto &&rec_mscope = recursive_move_scope();
    for(const move_t m : mline) {
      assert(!is_draw() && can_move());
      rec_mscope.scope(m);
    }
    return is_draw() || !can_move();
  }

  // methods that eventually generate completely legal moves
  // this method can be used on every iteration, but in fact is used on initialization only
  ALWAYS_UNROLL void init_state_attacks() {
    //for(auto&a:state.attacks)a=0ULL;
    memset(state.attacks.data(), 0x00, sizeof(piece_bitboard_t)*board::SIZE);
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

  // sub-routines for get_attacks_to
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

  // used by SEE, so must be very efficient yet flexible
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

  // update attack map when a move is made, for each affected position
  // this works significanltly faster than init_state_attacks
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

  // implementation of get_attack_mask
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

  // checkline: two bitboards, which filter possible moves by pieces other than king
  // e.g. all-ones when no check, knight when it is checking, 0 when double check
  void init_state_checkline(COLOR c) {
    assert(c < NO_COLORS);
    const piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    const piece_bitboard_t attackers = get_attacks_to(pos_king[c], enemy_of(c), occupied);
    // not in check
    if(!attackers) {
      state.checkline[c] = ~0x00ULL;
      return;
    } else if(!bitmask::is_exp2(attackers)) {
      // double-check
      state.checkline[c] = 0x00ULL;
      return;
    }
    // definitely only one attacker
    const pos_t attacker = bitmask::log2_of_exp2(attackers);
    state.checkline[c] = 0x00;
    if(bits_slid_diag & attackers) {
      state.checkline[c] |= piece::get_sliding_diag_attacking_ray(pos_king[c], attacker, occupied & ~piece::pos_mask(pos_king[c]));
    }
    if(bits_slid_orth & attackers) {
      state.checkline[c] |= piece::get_sliding_orth_attacking_ray(pos_king[c], attacker, occupied & ~piece::pos_mask(pos_king[c]));
    }
    // if pawn/knight, there is no ray, just the attacking piece that can be captured
    state.checkline[c] |= attackers;
  }

  // sub-routines to evaluate game state: draw, mate, etc
  // 50-halfmoves-draw
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

  // in crazyhouse, can still drop moves when no normal moves are available
  INLINE bool can_drop_move() const {
    if(!crazyhouse)return false;
    const COLOR c = activePlayer();
    // have to drop on checkline: if in check, cover the king
    const piece_bitboard_t ch_drop_locations = ~(bits[WHITE]|bits[BLACK]) & state.checkline[c];
    if(!ch_drop_locations)return false;
    // got any pieces to drop? is there anywhere to drop?
    // can't drop pawns at back ranks
    if(n_subs[Piece::get_piece_index(PAWN, c)] && ch_drop_locations & board::PAWN_RANKS) {
      return true;
    }
    for(PIECE p : {KNIGHT, BISHOP, ROOK, QUEEN}) {
      if(n_subs[Piece::get_piece_index(p, c)])return true;
    }
    return false;
  }

  INLINE bool can_move() const {
    assert(state.moves_initialized);
    const COLOR c = activePlayer();
    bool canmove = false;
    bitmask::foreach_early_stop(bits[c], [&](pos_t pos) mutable -> bool {
      if(state.moves[pos])canmove=true;
      return !canmove;
    });
    return canmove || can_drop_move();
  }

  INLINE bool is_draw_stalemate() const {
    const COLOR c = activePlayer();
    return !get_attacks_to(pos_king[c], enemy_of(c), bits[WHITE]|bits[BLACK]) && !can_move();
  }

  INLINE bool is_draw_material() const {
    if(bits_slid_orth)return false;
    const size_t no_pieces = piece::size(bits[WHITE] | bits[BLACK]);
    if(crazyhouse) {
      return no_pieces == 2 && !can_drop_move();
    }
    const piece_bitboard_t bishops = bits_slid_diag & ~bits_slid_orth,
                           knights = get_knight_bits();
    const piece_bitboard_t light_pieces = knights | bishops;
    return (no_pieces == 2)
        || (no_pieces == 3 && piece::size(light_pieces) == 1)
        || (no_pieces == 4
            && piece::size(light_pieces & bits[WHITE]) == 1
            && piece::size(light_pieces & bits[BLACK]) == 1);
  }

  INLINE bool is_draw_nogenmoves() const {
    return is_draw_halfmoves() || is_draw_material() || is_draw_repetition();
  }

  INLINE bool is_draw_with_genmoves() const {
    return is_draw_stalemate();
  }

  INLINE bool is_draw_() const {
    return is_draw_halfmoves() || is_draw_material() || is_draw_stalemate() || is_draw_repetition();
  }

  INLINE bool is_draw() const {
    assert(((is_draw_nogenmoves() || is_draw_with_genmoves()) ? 1 : 0) == (is_draw_() ? 1 : 0));
    return is_draw_();
  }

  INLINE void update_state_repetitions() {
    if(state_hist_repetitions <= self.get_current_ply()) {
      state_hist_repetitions = get_current_ply();
      return;
    }
    int repetitions = 1;
    const size_t no_iter = !crazyhouse ? std::min<size_t>(state_hist.size(), get_halfmoves())
                                       : state_hist.size();
    for(size_t i = NO_COLORS - 1; i < no_iter; i += NO_COLORS) {
      const auto &state_iter = state_hist[state_hist.size() - i - 1];
      if(state.info == state_iter.info && !state_iter.null_move_state) {
        ++repetitions;
        if(repetitions >= 3) {
          state_hist_repetitions = get_current_ply();
          return;
        }
      }
    }
    return;
  }

  INLINE bool is_draw_repetition() const {
    return state_hist_repetitions <= self.get_current_ply();
  }

  INLINE bool can_skip_genmoves() const {
    const COLOR c = activePlayer();
    const piece_bitboard_t kingcells = state.attacks[pos_king[c]] & ~bits[c];
    // - get_attack_mask ignores enemy king's presence
    return (get_attack_mask(enemy_of(c)) & kingcells) != kingcells;
  }

  INLINE void clear_state_unfinalize() {
    state.moves_initialized = false;
    for(auto&m:state.moves)m=0x00;
    state.pins = {0x00,0x00};
  }

  // move-generation, this is the main reason attack-generation and such exist
  void init_state_moves() {
    state.moves_initialized = true;
    assert(std::all_of(state.moves.begin(), state.moves.end(), [](auto m)->bool{return m==0x00;}));
    //if(is_draw_halfmoves()||is_draw_material())return;
//    for(const COLOR c : {WHITE, BLACK}) {
    const COLOR c = activePlayer(); {
      const COLOR ec = enemy_of(c);
      const piece_bitboard_t friends = bits[c], foes = bits[ec],
                             attack_mask = get_attack_mask(ec);
      const bool doublecheck = (state.checkline[c] == 0x00);
      if(!doublecheck) {
        const pos_t etrace = enpassant_trace();
        piece_bitboard_t foes_pawn = foes;
        if(etrace != board::nopos && c == activePlayer()) {
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
                                                                   qcastlrook[c], kcastlrook[c], chess960);
      }
    }
    init_state_moves_checkline_enpassant_takes();
    init_state_pins();
    if(chess960) {
//      for(const COLOR c : {WHITE, BLACK}) {
      const COLOR c = activePlayer(); {
        state.moves[pos_king[c]] &= ~state.pins[c];
      }
    }
    state.moves_initialized = true;
  }

  // sometimes, enpassant taking of a nearby pawn opens a line for a rook/queen
  // which potentially captures the king. this is the only case in which x-ray checks
  // are insufficient
  // if so, add pawns that can take checking en-passant to checkline
  void init_state_moves_checkline_enpassant_takes() {
    const COLOR c = activePlayer();
    const pos_t etrace = enpassant_trace();
    const pos_t epawn = enpassant_pawn();
    if(etrace == board::nopos)return;
    if(!bitmask::is_exp2(state.checkline[c]))return;
    const pos_t attacker = bitmask::log2_of_exp2(state.checkline[c]);
    if(epawn != attacker)return;
    const piece_bitboard_t apawns = get_pawn_attacks_to(etrace,c);
    if(!apawns)return;
    bitmask::foreach(apawns, [&](pos_t apawn) mutable noexcept -> void {
      state.moves[apawn] |= piece::pos_mask(etrace);
    });
  }

  // x-ray checks
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

  // two bitboards, each showing pinned pieces positions
  void init_state_pins() {
    // for efficiency can avoid the loop
    for(COLOR c : {WHITE, BLACK}) {
//    const COLOR c = activePlayer(); {
      //state.pins[c] = 0x00ULL;
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

  // non-xray pin generated in the case when en-passant can't be captured
  // k * * p P * * (R|Q)
  void init_horizontal_enpassant_pin() {
    if(enpassant_trace() == board::nopos)return;
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    const pos_t r = (c == WHITE) ? -1+5 : -1+4;
    if(
        state.checkline[c] != bitmask::full
        || !(piece::rank_mask(r) & bits[ec] & bits_slid_orth))
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

  // methods to print and debug stuff
  NEVER_INLINE void print() const {
    std::cout << "Active player: " << (activePlayer() == WHITE ? "WHITE" : "BLACK") << std::endl;
    for(pos_t i = board::LEN; i > 0; --i) {
      for(pos_t j = 0; j < board::LEN; ++j) {
        const Piece p = self[(i-1) * board::LEN + j];
        std::cout << p.str() << " ";
      }
      std::cout << std::endl;
    }
    if(crazyhouse) {
      std::cout << "Drop pieces" << std::endl;
      for(pos_t i = 0; i < board::NO_DROPPIECE_INDICES; ++i) {
        std::cout << pieces[i].str() << " ";
      }
      std::cout << std::endl;
      std::cout << str::join(n_subs, " ") << std::endl;
      if(bits_promoted_pawns) {
        std::cout << "Promoted pieces" << std::endl;
        bitmask::foreach(bits_promoted_pawns, [&](pos_t pos) mutable noexcept -> void {
          std::cout << board::_pos_str(pos) << "=" << self[pos].str() << " ";
        });
        std::cout << std::endl;
      }
    }
    str::print("FEN:", fen::export_as_string(export_as_fen()));
  }

  NEVER_INLINE std::string _move_str(const move_t m) const {
    if(m==board::nullmove)return "0000"s;
    const pos_t _i = bitmask::first(m) & board::MOVEMASK;
    return board::_move_str(m, bits_pawns & piece::pos_mask(_i));
  }

  NEVER_INLINE std::vector<std::string> _line_str(const MoveLine &line, bool thorough=true) {
    std::vector<std::string> s;
    if(thorough)assert(check_valid_sequence(line));
    decltype(auto) rec_mscope = self.recursive_move_unfinalized_scope();
    for(const auto m : line) {
      if(m==board::nullmove)break;
      s.emplace_back(_move_str(m));
      if(thorough) {
        rec_mscope.scope(m);
      }
    }
    return s;
  }

  NEVER_INLINE std::string _line_str_full(const MoveLine &line, bool thorough_fut=true) {
    std::string s = "["s + str::join(_line_str(line.get_past(), false), " "s) + "]"s;
    if(!line.empty()) {
      s += " "s + str::join(_line_str(line.get_future(), thorough_fut), " "s);
    }
    return s;
  }

  board_info get_info_from_line(MoveLine &mline) {
    assert(check_valid_sequence(mline));
    decltype(auto) rec_mscope = self.recursive_move_unfinalized_scope();
    for(const move_t &m : mline) {
      rec_mscope.scope(m);
    }
    return state.info;
  }

  fen::FEN export_as_fen() const {
    fen::FEN f = {
      .active_player=activePlayer(),
      .board=""s,
      .subs=""s,
      .castlings=get_castlings_rook_mask(),
      .enpassant=enpassant_trace(),
      .halfmove_clock=get_halfmoves(),
      .fullmove=ply_index_t(((get_current_ply() - 1) / 2) + 1),
      .chess960=chess960,
      .crazyhouse=crazyhouse,
      .crazyhouse_promoted=0x00,
    };
    pos_t write_index = 0;
    for(pos_t y = 0; y < board::LEN; ++y) {
      for(pos_t x = 0; x < board::LEN; ++x, ++write_index) {
        const pos_t ind = board::_pos(A+x, 8-y);
        if(self.empty_at_pos(ind)) {
          f.board += ' ';
        } else {
          f.board += self[ind].str();
        }
        // in crazyhouse, set promoted pawns
        if(f.crazyhouse && piece::is_set(bits_promoted_pawns, ind)) {
          piece::set_pos(f.crazyhouse_promoted, write_index);
        }
      }
    }
    if(f.crazyhouse) {
      for(COLOR c : {WHITE, BLACK}) {
        for(PIECE p : {QUEEN,ROOK,BISHOP,KNIGHT,PAWN}) {
          Piece piece = Piece(p, c);
          for(pos_t i = 0; i < n_subs[piece.piece_index]; ++i) {
            f.subs += piece.str();
          }
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
