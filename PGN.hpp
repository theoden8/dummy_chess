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
  std::vector<std::string> ply;
  std::string ending = "";

  PGN(Board &board):
    board(board)
  {
    startfen = board.export_as_fen();
  }

  size_t size() const {
    return cur_ply;
  }

  char name_of_file(pos_t i) const {
    return 'a' + board::_x(i);
  }

  char name_of_rank(pos_t i) const {
    return '1' + board::_y(i);
  }

  std::string piece_name(Piece p) {
    std::string s = ""s;
    if(p.value == PAWN)return s;
    s += toupper(p.str());
    return s;
  }

  std::string resolve_ambiguity(pos_t i, pos_t j, bool enpassant=false) const {
    std::string resolve = ""s;
    piece_bitboard_t imask = ~0uLL,
                     jmask = 1ULL << j;
    if(board[i].value == PAWN && (board[i].value == PAWN || board[i].value == EMPTY)) {
      if(board[j].value != EMPTY || enpassant) {
        resolve += name_of_file(i);
      }
      imask = board::file_mask(board::_x(i));
      if(!LICHESS_COMPATIBILITY) {
        jmask = board::file_mask(board::_x(j));
      }
    }
    bool file_resolved = false, rank_resolved = false;

    piece_bitboard_t mask = ~0x00ULL;
    const COLOR c = board[i].color;
    if(board.bits_pawns & piece::pos_mask(i)) {
      mask = board.bits_pawns;
    } else if(board.bits_slid_diag & board.bits_slid_orth & piece::pos_mask(i)) {
      mask = board.bits_slid_diag & board.bits_slid_orth;
    } else if(board.bits_slid_diag & piece::pos_mask(i)) {
      mask = board.bits_slid_diag & ~board.bits_slid_orth;
    } else if(board.bits_slid_orth & piece::pos_mask(i)) {
      mask = board.bits_slid_orth & ~board.bits_slid_diag;
    } else if(i == board.pos_king[c]) {
      mask = piece::pos_mask(i);
    } else {
      mask = board.get_knight_bits();
    }
    mask &= board.bits[c];

    bitmask::foreach(mask, [&](pos_t k) mutable -> void {
      if(rank_resolved && file_resolved)return;
      if(!(imask & (1ULL << k)))return;
      if(board.state.moves[k] & jmask) {
        if(!file_resolved && board::_x(k) != board::_x(i)) {
          resolve += name_of_file(i);
          file_resolved = true;
        } else if(!rank_resolved && board::_y(k) != board::_y(i)) {
          resolve += name_of_rank(i);
          rank_resolved = true;
        }
      }
    });
    return resolve;
  }

  void write_move(pos_t i, pos_t j) {
    const move_t m = bitmask::_pos_pair(i, j);
    const pos_t promote_as = j & ~board::MOVEMASK;
    j &= board::MOVEMASK;
    std::string p;
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
      const pos_t killwhere = board.enpassant_pawn();
      p += resolve_ambiguity(i, j, true);
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
      if(board[i].value==PAWN && board[j].value==PAWN && is_capture && !LICHESS_COMPATIBILITY) {
        p += resolve_ambiguity(i, j);
        p += name_of_file(j);
      } else {
        p += piece_name(board[i]);
        p += resolve_ambiguity(i, j);
        if(is_capture)p+='x';
        p += board::_pos_str(j);
      }
    }
    ++cur_ply;
    ply.emplace_back(p);
  }

  INLINE void write_move(move_t m) {
    write_move(bitmask::first(m), bitmask::second(m));
  }

  void handle_move(move_t m) {
    assert(m == board::nullmove || board.check_valid_move(m));
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
  }

  void handle_move(pos_t i, pos_t j) {
    handle_move(bitmask::_pos_pair(i, j));
  }

  void read_move(const std::string &s) {
    assert(!s.empty());
    COLOR c = board.activePlayer();
    // castlings
    if(s == "O-O"s || s == "O-O+"s || s == "O-O#") {
      // short castling
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      pos_t k_move_to = board::_pos(G, castlrank);
      if(board.chess960) {
        k_move_to = board::_pos(board.kcastlrook[c], castlrank);
      }
      handle_move(bitmask::_pos_pair(board.pos_king[c], k_move_to));
      return;
    } else if(s == "O-O-O"s || s == "O-O-O+"s || s == "O-O-O#"s) {
      // long castling
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      pos_t k_move_to = board::_pos(C, castlrank);
      if(board.chess960) {
        k_move_to = board::_pos(board.qcastlrook[c], castlrank);
      }
      handle_move(bitmask::_pos_pair(board.pos_king[c], k_move_to));
      return;
    }
    long long i = 0, j = s.length() - 1;
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
    bool assert_capture = false, assert_check = false, assert_mate = false;
    if(s[j] == '+') {
      assert_check = true;
      --j;
    } else if(s[j] == '#') {
      assert_check = true, assert_mate = true;
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
    assert(index("123456789", s[j]) != nullptr);
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
    if(s[j] == 'x') {
      assert_capture = true;
      --j;
    }
    bool capture_verified = !board.empty_at_pos(move_to & board::MOVEMASK);
    if(p == PAWN && (move_to & board::MOVEMASK) == board.enpassant_trace()) {
      capture_verified = true;
    }
    assert((assert_capture && capture_verified) || (!assert_capture && !capture_verified));
    pos_t move_from = board::nopos;
    if(s[j] == '@') {
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
        if(board.state.moves[from_pos] & piece::pos_mask(move_to)) {
          filtered |= piece::pos_mask(from_pos);
        }
      });
      from_positions = filtered;
      if(!bitmask::is_exp2(from_positions)) {
        abort();
      }
      move_from = bitmask::log2_of_exp2(from_positions);
    }
    const move_t m = bitmask::_pos_pair(move_from, move_to);
    handle_move(m);
    c = board.activePlayer();
    const bool check_verified = (board.state.checkline[c] != bitmask::full);
    assert((assert_check && check_verified) || (!assert_check && !check_verified));
    const bool mate_verified = board.is_checkmate();
    assert((assert_mate && mate_verified) || (!assert_mate && !mate_verified));
    assert(ply.back() == s);
  }

  void read(const std::string &s) {
    size_t i = 0;
    while(i < s.length()) {
//      char sss[] = {s[i], '\0'};
//      str::print("character:", sss, i);
      if(isspace(s[i])) {

        ++i;
        continue;
      } else if(s[i] == '[') {
        i = s.find("]", i + 1) + 1;
        if(i == std::string::npos)break;
        continue;
      } else if(s[i] == '{') {
        i = s.find("}", i + 1) + 1;
        if(i == std::string::npos)break;
        continue;
      } else if(s[i] == '(') {
        i = s.find(")", i + 1) + 1;
        if(i == std::string::npos)break;
        continue;
      } else if(index("0123456789.", s[i]) != nullptr) {
        ++i;
        continue;
      }
      size_t j = s.find(" ", i);
      if(j == std::string::npos) {
        break;
      }
      const std::string ss = s.substr(i, j-i);
//      str::print("new read:", "<"s + ss + ">"s);
      read_move(ss);
      i = j;
    }
  }

  void retract_move() {
    if(cur_ply != 0) {
      board.retract_move();
      --cur_ply;
      ply.pop_back();
      ending = ""s;
    }
  }

  NEVER_INLINE std::string str() const {
    std::string s;
    if(startfen != fen::starting_pos) {
      s += "[FEN] \""s + fen::export_as_string(startfen) + "\"]\n\n";
      s += "[FEN] \""s + fen::export_as_string(fen::starting_pos) + "\"]\n\n";
    }
    for(size_t i = 0; i < cur_ply; ++i) {
      if(!(i & 1)) {
        s += " "s + std::to_string(i / 2 + 1) + "."s;
      }
      s += " "s + ply[i];
    }
    return s;
  }
};

std::string _move_str(Board &b, move_t m) {
  assert(m == board::nullmove || b.check_valid_move(m));
  pgn::PGN pgn(b);
  pgn.handle_move(m);
  std::string s = pgn.ply.front();
  pgn.retract_move();
  return s;
}

NEVER_INLINE std::string _line_str(Board &b, const MoveLine &mline) {
  assert(b.check_valid_sequence(mline));
  pgn::PGN pgn(b);
  for(auto m : mline) {
    pgn.handle_move(m);
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
  for(auto m : mline.get_past()) {
    b.retract_move();
  }
  for(auto m : mline.get_past()) {
    pgn.handle_move(m);
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
  std::string s; char c;
  while((c=fgetc(fp))!=EOF)s+=c;
  fclose(fp);
  return load_from_string(s, board);
}

} // namespace pgn

NEVER_INLINE std::string MoveLine::pgn(Board &b) const {
  return pgn::_line_str(b, *this);
}

NEVER_INLINE std::string MoveLine::pgn_full(Board &b) const {
  return pgn::_line_str_full(b, *this);
}
