#pragma once


#include <string>
#include <vector>

#include <Board.hpp>
#include <MoveLine.hpp>


namespace pgn {

constexpr bool LICHESS_COMPATIBILITY = 1;

struct PGN {
  fen::FEN startfen;
  Board &board;
  size_t cur_ply = 0;
  EXPORT std::vector<std::string> ply;
  EXPORT std::vector<std::string> comments;  // Comment after each move (may contain [%eval])
  std::string ending = "";

  EXPORT explicit PGN(Board &board):
    board(board)
  {
    startfen = board.export_as_fen();
  }

  EXPORT size_t size() const {
    return cur_ply;
  }

  char name_of_file(pos_t i) const {
    return 'a' + board::_x(i);
  }

  char name_of_rank(pos_t i) const {
    return '1' + board::_y(i);
  }

  // coordinate specification when move is ambiguous.
  // Rd1 is ambiguous, so use R[a]d1 or R[f]d1, or R[5]d1, potentially both could be used
  std::string resolve_ambiguity(pos_t i, pos_t j, bool enpassant=false) const {
    const bool finalized = board.state.moves_initialized;
    if(!finalized)board.make_move_finalize();
    // drop information about promotions:
    j &= board::MOVEMASK;
    std::string resolve = ""s;
    // ambiguity masks for source and destination piece
    piece_bitboard_t imask = bitmask::full,
                     jmask = piece::pos_mask(j);
    // each file and each rank should occur only once
    bool file_resolved = false, rank_resolved = false;
    if(piece::is_set(board.bits_pawns, i)) {
      if(board.empty_at_pos(j) && !enpassant) {
        // pawn push: []d4
        imask = piece::file_mask(board::_x(i));
      } else {
        // [d]xc4
        resolve += name_of_file(board::_x(i));
        file_resolved = true;
        const pos_t jfile = board::_x(j);
        if(jfile == A) {
          imask = piece::file_mask(B);
        } else if(jfile == H) {
          imask = piece::file_mask(G);
        } else {
          imask = piece::file_mask(jfile - 1) | piece::file_mask(jfile + 1);
        }
      }
      // restrict source mask to the file
      // dc vs dxc4
      // in the latter case,
      if(!LICHESS_COMPATIBILITY) {
        jmask = board::file_mask(board::_x(j));
      }
      if(!finalized)board.clear_state_unfinalize();
      return resolve;
    }

    // mask for all piece types and color as the i-piece
    piece_bitboard_t mask = bitmask::full;
    const COLOR c = board[i].color;
    if(piece::is_set(board.bits_pawns, i)) {
      mask = board.bits_pawns;
    } else if(piece::is_set(board.bits_slid_diag & board.bits_slid_orth, i)) {
      mask = board.bits_slid_diag & board.bits_slid_orth;
    } else if(piece::is_set(board.bits_slid_diag, i)) {
      mask = board.bits_slid_diag & ~board.bits_slid_orth;
    } else if(piece::is_set(board.bits_slid_orth, i)) {
      mask = board.bits_slid_orth & ~board.bits_slid_diag;
    } else if(i == board.pos_king[c]) {
      mask = piece::pos_mask(i);
    } else {
      mask = board.get_knight_bits();
    }
    mask &= board.bits[c];

    // search the mask of possible source pieces, and resolve ambiguity if there is any
    bitmask::foreach(mask & imask, [&](pos_t i_candidate) mutable -> void {
      if(rank_resolved && file_resolved)return;
      // check if candidate can reach destination
      if(board.state.moves[i_candidate] & jmask) {
        if(!file_resolved && board::_x(i_candidate) != board::_x(i)) {
          resolve += name_of_file(i);
          file_resolved = true;
        } else if(!rank_resolved && board::_y(i_candidate) != board::_y(i)) {
          resolve += name_of_rank(i);
          rank_resolved = true;
        }
      }
    });
    return resolve;
  }

  void write_move(pos_t i, pos_t j) {
    const move_t m = bitmask::_pos_pair(i, j);
    // split promotion and destination square information
    const pos_t promote_as = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    std::string p;
    // consider the type of move
    if(m == board::nullmove) {
      p = "0000"s;
    } else if(board.is_drop_move(i, j)) {
      p = board::_move_str(bitmask::_pos_pair(i, j));
    } else if(board.is_castling_move(i, j)) {
      const COLOR c = board.color_at_pos(i);
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      pos_t k_j = j;
      if(board.chess960) {
        if(j == board::_pos(board.qcastlrook[c], castlrank)) {
          k_j = board::_pos(C, castlrank);
        } else if(j == board::_pos(board.kcastlrook[c], castlrank)) {
          k_j = board::_pos(G, castlrank);
        } else {
          abort();
        }
      }
      const move_t rookmove = piece::get_king_castle_rook_move(c, i, k_j, board.qcastlrook[c], board.kcastlrook[c]);
      const pos_t r_i = bitmask::first(rookmove),
                  r_j = bitmask::second(rookmove);
      if(board::_x(r_i) == board.qcastlrook[c]) {
        p = "O-O-O"s;
      } else {
        p = "O-O"s;
      }
    } else if(board.is_enpassant_take_move(i, j)) {
      //const pos_t killwhere = board.enpassant_pawn();
      p += resolve_ambiguity(i, j, true);
      // dc vs dxc4
      if(LICHESS_COMPATIBILITY) {
        p += 'x';
        p += board::_pos_str(j);
      } else {
        p += name_of_file(j);
      }
    } else if(board.is_promotion_move(i, j)) {
      const PIECE becomewhat = board::get_promotion_as(promote_as);
      const bool is_capture = !board.empty_at_pos(j);
      if(is_capture) {
        p += resolve_ambiguity(i, j);
        p += 'x';
      }
      p += board::_pos_str(j);
      p += '=';
      p += toupper(Piece(becomewhat, board.activePlayer()).str());
    } else {
      const bool is_capture = !board.empty_at_pos(j);
      p = "";
      // pawn takes pawn (not en-passant)
      // lichess compatibiltiy: dc vs dxc4
      if(board[i].value==PAWN && board[j].value==PAWN && is_capture && !LICHESS_COMPATIBILITY) {
        // example: dc
        p += resolve_ambiguity(i, j);
        p += name_of_file(j);
      } else {
        // P [ambiguity] [x] c4
        if(~board.bits_pawns & piece::pos_mask(i)) {
          p += toupper(board[i].str());
        }
        p += resolve_ambiguity(i, j);
        if(is_capture)p+='x';
        p += board::_pos_str(j);
      }
    }
    ++cur_ply;
    ply.emplace_back(p);
    comments.emplace_back("");  // Empty comment placeholder
  }

  INLINE void write_move(move_t m) {
    write_move(bitmask::first(m), bitmask::second(m));
  }

  // write move, advance board state and determine game-state
  // Returns false if move is invalid (corrupted PGN)
  // If strict=true, asserts instead of returning false
  bool handle_move(move_t m, bool strict) {
    if (!board.check_valid_move(m, false)) {
      if (strict) {
        assert(false && "Invalid move in PGN");
      }
      return false;
    }
    write_move(m);
    board.make_move(m);
    const COLOR c = board.activePlayer();
    const pos_t no_checks = board.get_attack_counts_to(board.pos_king[c], enemy_of(c));
    ending = "";
    if(board.is_draw_stalemate()) {
      ending = "1/2 - 1/2 (stalemate)";
    } else if(board.is_draw_halfmoves()) {
      ending = "1/2 - 1/2 (50 moves)";
    } else if(board.is_draw_material()) {
      ending = "1/2 - 1/2 (material)";
    } else if(board.is_checkmate()) {
      ply.back() += '#';
      ending = (c == WHITE) ? "1-0"s : "0-1"s;
    } else if(no_checks > 0) {
      ply.back()+='+';
      if(no_checks>1)ply.back()+='+';
    } else if(board.is_draw_repetition()) {
      ending = "1/2 - 1/2 (repetitions)";
    }
    return true;
  }

  bool handle_move(pos_t i, pos_t j, bool strict) {
    return handle_move(bitmask::_pos_pair(i, j), strict);
  }

  void handle_line(MoveLine &mline, bool strict) {
    for(const move_t &m : mline) {
      handle_move(m, strict);
    }
  }

  // read and return move, return whether check or mate are explicitly annotated
  EXPORT move_t read_move_with_flags(const std::string &s, bool &flag_check, bool &flag_mate) {
    assert(!s.empty());
    COLOR c = board.activePlayer();
    // castlings
    long long j = s.length() - 1;
    // (O-O|O-O-O) [+|#]
    if(s == "O-O"s || s == "O-O+"s || s == "O-O#") {
      // short castling
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      pos_t k_move_to = board::_pos(G, castlrank);
      if(board.chess960) {
        k_move_to = board::_pos(board.kcastlrook[c], castlrank);
      }
      if(s[j] == '+') {
        flag_check = true;
      } else if(s[j] == '#') {
        flag_check = true, flag_mate = true;
      }
      return bitmask::_pos_pair(board.pos_king[c], k_move_to);
    } else if(s == "O-O-O"s || s == "O-O-O+"s || s == "O-O-O#"s) {
      // long castling
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      pos_t k_move_to = board::_pos(C, castlrank);
      if(board.chess960) {
        k_move_to = board::_pos(board.qcastlrook[c], castlrank);
      }
      if(s[j] == '+') {
        flag_check = true;
      } else if(s[j] == '#') {
        flag_check = true, flag_mate = true;
      }
      return bitmask::_pos_pair(board.pos_king[c], k_move_to);
    }
    long long i = 0;
    // [P] [ @ | [F][R][x] | [R][F][x] ] FR [=P] [+|#]
    // first, read piece
    PIECE p = PAWN;
    if(isupper(s[i])) {
      // Nxd4 N@d5 Bg6
      // determine piece type, in case this is a drop move
      switch(s[i]) {
        case 'P':p=PAWN;break;
        case 'N':p=KNIGHT;break;
        case 'B':p=BISHOP;break;
        case 'R':p=ROOK;break;
        case 'Q':p=QUEEN;break;
        case 'K':p=KING;break;
        default:abort();
      }
      //++i;
    }
    // second, read destination
    bool assert_capture = false;
    if(s[j] == '+') {
      flag_check = true;
      --j;
    } else if(s[j] == '#') {
      flag_check = true, flag_mate = true;
      --j;
    }
    PIECE move_promotion_as = PAWN;
    if(index("NBRQ", s[j]) != nullptr) {
      switch(s[j]) {
        case 'N':move_promotion_as=KNIGHT;break;
        case 'B':move_promotion_as=BISHOP;break;
        case 'R':move_promotion_as=ROOK;break;
        case 'Q':move_promotion_as=QUEEN;break;
        default:abort();
      }
      --j;
      assert(s[j] == '=');
      --j;
    }
    assert(index("12345678", s[j]) != nullptr);
    const pos_t to_rank = s[j] - '1';
    --j;
    assert(index("abcdefgh", s[j]) != nullptr);
    const pos_t to_file = s[j] - 'a';
    pos_t move_to = board::_pos(A+to_file, 1+to_rank);
    switch(move_promotion_as) {
      case KNIGHT:move_to|=board::PROMOTE_KNIGHT;break;
      case BISHOP:move_to|=board::PROMOTE_BISHOP;break;
      case ROOK:move_to|=board::PROMOTE_ROOK;break;
      case QUEEN:move_to|=board::PROMOTE_QUEEN;break;
      default:break;
    }
    --j;
    if(j >= 0 && s[j] == 'x') {
      assert_capture = true;
      --j;
    }
    bool capture_verified = !board.empty_at_pos(move_to & board::MOVEMASK);
    if(p == PAWN && (move_to & board::MOVEMASK) == board.enpassant_trace()) {
      capture_verified = true;
    }
    assert((assert_capture && capture_verified) || (!assert_capture && !capture_verified));
    pos_t move_from = board::nopos;
    if(j >= 0 && s[j] == '@') {
      move_from = board::CRAZYHOUSE_DROP | pos_t(p);
    } else {
      piece_bitboard_t from_positions = bitmask::full;
      assert(from_positions);
      switch(p) {
        case PAWN:
          from_positions = board.bits_pawns;
        break;
        case KNIGHT:
          from_positions = board.get_knight_bits();
        break;
        case BISHOP:
          from_positions = ~board.bits_slid_orth & board.bits_slid_diag;
        break;
        case ROOK:
          from_positions = board.bits_slid_orth & ~board.bits_slid_diag;
        break;
        case QUEEN:
          from_positions = board.bits_slid_orth & board.bits_slid_diag;
        break;
        case KING:
          from_positions = board.get_king_bits();
        break;
        default:abort();
      }
      from_positions &= board.bits[board.activePlayer()];
      if(j >= i && index("12345678", s[j]) != nullptr) {
        from_positions &= piece::rank_mask(s[j] - '1');
        --j;
        if(j >= i && index("abcdefgh", s[j]) != nullptr) {
          from_positions &= piece::file_mask(s[j] - 'a');
          --j;
        }
      } else if(j >= i && index("abcdefgh", s[j]) != nullptr) {
        from_positions &= piece::file_mask(s[j] - 'a');
        --j;
        if(j >= i && index("12345678", s[j]) != nullptr) {
          from_positions &= piece::rank_mask(s[j] - '1');
          --j;
        }
      }
      piece_bitboard_t filtered = 0x00;
      bitmask::foreach(from_positions, [&](pos_t from_pos) mutable noexcept -> void {
        if(board.state.moves[from_pos] & piece::pos_mask(move_to & board::MOVEMASK)) {
          filtered |= piece::pos_mask(from_pos);
        }
      });
      from_positions = filtered;
      if(!bitmask::is_exp2(from_positions)) {
        abort();
      }
      move_from = bitmask::log2_of_exp2(from_positions);
    }
    return bitmask::_pos_pair(move_from, move_to);
  }

  // perform move, verify check/mate/capture situation
  // Returns false if move is invalid (corrupted PGN)
  // If strict=true, asserts instead of returning false
  EXPORT bool read_move(const std::string &s, bool strict=false) {
    // Disable draw claims during PGN parsing - games can continue past unclaimed draws
    const bool _repetition_is_draw = board.repetition_is_draw;
    const bool _halfmoves_is_draw = board.halfmoves_is_draw;
    board.repetition_is_draw = false;
    board.halfmoves_is_draw = false;
    bool assert_check = false, assert_mate = false;
    const move_t m = read_move_with_flags(s, assert_check, assert_mate);
    if (!handle_move(m, strict)) {
      board.repetition_is_draw = _repetition_is_draw;
      board.halfmoves_is_draw = _halfmoves_is_draw;
      return false;
    }
    const COLOR c = board.activePlayer();
    const bool check_verified = (board.state.checkline[c] != bitmask::full);
    // If PGN claims check/mate, verify it's true (return false if not)
    if (assert_check && !check_verified) {
      if (strict) {
        assert(false && "PGN claims check but position is not in check");
      }
      board.repetition_is_draw = _repetition_is_draw;
      board.halfmoves_is_draw = _halfmoves_is_draw;
      return false;
    }
    const bool mate_verified = board.is_checkmate();
    if (assert_mate && !mate_verified) {
      if (strict) {
        assert(false && "PGN claims checkmate but position is not checkmate");
      }
      board.repetition_is_draw = _repetition_is_draw;
      board.halfmoves_is_draw = _halfmoves_is_draw;
      return false;
    }
    // Note: ply.back() may differ from s due to disambiguation format
    // e.g. input "R1d2" vs write_move generates "Rd1d2"
    board.repetition_is_draw = _repetition_is_draw;
    board.halfmoves_is_draw = _halfmoves_is_draw;
    return true;
  }

  // Parse a header line like [Tag "Value"], return the value for given tag or empty
  static std::string parse_header(const std::string &header, const std::string &tag) {
    size_t tag_pos = header.find(tag + " \"");
    if (tag_pos == std::string::npos) return "";
    size_t val_start = header.find('"', tag_pos) + 1;
    size_t val_end = header.find('"', val_start);
    if (val_start == 0 || val_end == std::string::npos) return "";
    return header.substr(val_start, val_end - val_start);
  }

  // read PGN with headers and comments
  // If store_comments is true, stores comments in the comments vector
  // Returns false if PGN contains invalid moves (corrupted game)
  // If strict=true, asserts instead of returning false
  bool read(const std::string &s, bool store_comments=false, bool strict=false) {
    size_t i = 0;
    bool in_moves = false;  // Have we seen any moves yet?

    while(i < s.length()) {
      if(isspace(s[i])) {
        ++i;
        continue;
      } else if(s[i] == '[') {
        // Header tag
        size_t end = s.find("]", i + 1);
        if(end == std::string::npos) break;

        if (!in_moves) {
          // Parse FEN header if present
          std::string header = s.substr(i + 1, end - i - 1);
          std::string fen_str = parse_header(header, "FEN");
          if (!fen_str.empty()) {
            startfen = fen::load_from_string(fen_str);
            board.set_fen(startfen);
          }
        }
        i = end + 1;
        continue;
      } else if(s[i] == '{') {
        // Comment
        size_t end = s.find("}", i + 1);
        if(end == std::string::npos) break;

        if (store_comments && !comments.empty()) {
          // Append to last move's comment
          std::string comment = s.substr(i + 1, end - i - 1);
          if (!comments.back().empty()) comments.back() += " ";
          comments.back() += comment;
        }
        i = end + 1;
        continue;
      } else if(s[i] == '(') {
        // Variation - skip with proper nesting
        int depth = 1;
        ++i;
        while(i < s.length() && depth > 0) {
          if(s[i] == '(') depth++;
          else if(s[i] == ')') depth--;
          ++i;
        }
        continue;
      } else if(s[i] == '$') {
        // NAG - skip
        ++i;
        while(i < s.length() && isdigit(s[i])) ++i;
        continue;
      } else if(index("0123456789.", s[i]) != nullptr) {
        // Move number or result - skip digits and dots
        // But check for game result first
        if (s.substr(i, 3) == "1-0" || s.substr(i, 3) == "0-1") {
          break;  // Game over
        }
        if (s.substr(i, 7) == "1/2-1/2") {
          break;  // Draw
        }
        ++i;
        continue;
      } else if(s[i] == '*') {
        // Unfinished game
        break;
      }

      // Find end of move token
      size_t j = i;
      while(j < s.length() && !isspace(s[j]) && s[j] != '{' && s[j] != '(' && s[j] != '$') {
        ++j;
      }
      if(j == i) {
        ++i;
        continue;
      }

      std::string ss = s.substr(i, j - i);

      // Strip annotations like ! ? !! ?? !? ?!
      while(!ss.empty() && (ss.back() == '!' || ss.back() == '?')) {
        ss.pop_back();
      }

      if(!ss.empty() && ss != "1-0" && ss != "0-1" && ss != "1/2-1/2" && ss != "*") {
        if (!read_move(ss, strict)) {
          return false;  // Invalid move - corrupted PGN
        }
        in_moves = true;
      }
      i = j;
    }
    return true;
  }

  void retract_move() {
    if(cur_ply != 0) {
      board.retract_move();
      --cur_ply;
      ply.pop_back();
      if (!comments.empty()) comments.pop_back();
      ending = ""s;
    }
  }

  EXPORT NEVER_INLINE std::string str() const {
    std::string s;
    if(startfen != fen::starting_pos) {
      if(!board.chess960 && !board.crazyhouse) {
        s += "[Variant \"From Position\"]\n";
      }
      s += "[FEN \""s + fen::export_as_string(startfen) + "\"]\n";
      s += "\n";
    } else {
      //s += "[FEN \""s + fen::export_as_string(fen::starting_pos) + "\"]\n\n";
    }
    for(size_t i = 0; i < cur_ply; ++i) {
      if(!(i & 1)) {
        s += " "s + std::to_string(i / 2 + 1) + "."s;
      }
      s += " "s + ply[i];
    }
    return s;
  }

  EXPORT static pgn::PGN load_from_string(const std::string &s, Board &board);
  EXPORT static pgn::PGN load_from_file(const std::string &fname, Board &board);
};

// functions to return as string and debug
std::string _move_str(Board &b, move_t m) {
  assert(m == board::nullmove || b.check_valid_move(m));
  pgn::PGN pgn(b);
  pgn.handle_move(m, true);
  std::string s = pgn.ply.front();
  pgn.retract_move();
  return s;
}

move_t _read_move(Board &b, const std::string &s) {
  pgn::PGN pgn(b);
  bool flag1, flag2;
  return pgn.read_move_with_flags(s, flag1, flag2);
}

NEVER_INLINE std::string _line_str(Board &b, const MoveLine &mline) {
  assert(b.check_valid_sequence(mline));
  pgn::PGN pgn(b);
  for(const move_t m : mline) {
    pgn.handle_move(m, true);
  }
  std::string s = str::join(pgn.ply, " "s);
  for(auto m : mline) {
    pgn.retract_move();
  }
  return s;
}

NEVER_INLINE std::string _line_str_full(Board &b, const MoveLine &mline) {
  assert(b.check_valid_sequence(mline));
  pgn::PGN pgn(b);
  for(const move_t m : mline.get_past()) {
    b.retract_move();
  }
  for(const move_t m : mline.get_past()) {
    pgn.handle_move(m, true);
  }
  std::string s = ""s;
  if(mline.start > 0) {
    s += "["s + str::join(pgn.ply, " "s) + "]"s;
    if(!mline.empty()) {
      s += " "s;
    }
  }
  if(!mline.empty()) {
    s += _line_str(b, mline.get_future());
  }
  return s;
}

pgn::PGN load_from_string(const std::string &s, Board &board) {
  pgn::PGN pgn(board);
  pgn.read(s);
  return pgn;
}

pgn::PGN load_from_file(const std::string &fname, Board &board) {
  FILE *fp = fopen(fname.c_str(), "r");
  assert(fp != nullptr);
  std::string s; int c;
  while((c=fgetc(fp))!=EOF)s+=c;
  fclose(fp);
  return load_from_string(s, board);
}

pgn::PGN PGN::load_from_string(const std::string &s, Board &board) {
  return pgn::load_from_string(s, board);
}

pgn::PGN PGN::load_from_file(const std::string &fname, Board &board) {
  return pgn::load_from_file(fname, board);
}

} // namespace pgn

// external methods
#define self (*this)
NEVER_INLINE std::string MoveLine::pgn(Board &b) const {
  return pgn::_line_str(b, self);
}

NEVER_INLINE std::string MoveLine::pgn_full(Board &b) const {
  return pgn::_line_str_full(b, self);
}
#undef self
