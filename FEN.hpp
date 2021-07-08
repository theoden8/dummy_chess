#pragma once


#include <cctype>
#include <strings.h>

#include <Bitboard.hpp>
#include <Piece.hpp>
#include <String.hpp>


class Board;

// shredder-fen
// https://www.chessprogramming.org/Forsyth-Edwards_Notation#Shredder-FEN
namespace fen {
  typedef struct _FEN {
    COLOR active_player : 2;
    std::string board;
    std::string subs;
    pos_pair_t castlings;
    pos_t enpassant;
    pos_t halfmove_clock;
    ply_index_t fullmove;
    bool traditional;
    bool crazyhouse;

    inline bool operator==(const struct _FEN &other) const {
      return active_player == other.active_player
             && board == other.board && subs == other.subs
             && castlings == other.castlings
             && enpassant == other.enpassant && halfmove_clock == other.halfmove_clock
             && fullmove == other.fullmove;
    }
  } FEN;

  FEN load_from_string(std::string s) {
    size_t i = 0;
    FEN f = {
      .active_player=WHITE,
      .board=""s,
      .subs=""s,
      .castlings=0x00,
      .enpassant=board::nopos,
      .halfmove_clock=0,
      .fullmove=0,
      .traditional=true,
      .crazyhouse=true,
    };
    // board
    pos_t kingpos[2] = {board::nopos, board::nopos};
    pos_t qrook[2] = {board::nopos, board::nopos};
    pos_t krook[2] = {board::nopos, board::nopos};
    pos_t board_index = 0;
//    str::pdebug("FEN", s);
    size_t slashcount = 0;
    for(const char *c = s.c_str(); *c != '\0' && *c != ' '; ++c, ++i) {
      assert(slashcount < 9);
      if(*c == '/') {
        ++slashcount;
        continue;
      }
      if(slashcount == 8) {
        f.crazyhouse = true;
        f.subs += *c;
      }
      if(isdigit(*c)) {
        for(int j=0;j<*c-'0';++j, ++board_index)f.board+=' ';
        continue;
      }
      const pos_t ind = board::_pos(board::_x(board_index), 8 - board::_y(board_index));
      f.board += *c;
      switch(*c) {
        case 'K':
        {
          kingpos[WHITE]=board::_x(ind);
//          const char ch[] = {char('A' + kingpos[WHITE]), '\0'};
//          str::pdebug("kingpos[WHITE]", ch);
        }
        break;
        case 'k':
        {
          kingpos[BLACK]=board::_x(ind);
//          const char ch[] = {char('A' + kingpos[BLACK]), '\0'};
//          str::pdebug("kingpos[BLACK]", ch);
        }
        break;
        case 'R':
        if(board::_y(ind) == -1+1) {
          if(kingpos[WHITE]==board::nopos) {
            qrook[WHITE] = board::_x(ind);
//            const char ch[] = {char('A' + qrook[WHITE]), '\0'};
//            str::pdebug("qrook[WHITE]", ch);
          } else {
            krook[WHITE] = board::_x(ind);
//            const char ch[] = {char('A' + krook[WHITE]), '\0'};
//            str::pdebug("krook[WHITE]", ch);
          }
        }
        break;
        case 'r':
        if(board::_y(ind) == -1+8) {
          if(kingpos[BLACK]==board::nopos) {
            qrook[BLACK] = board::_x(ind);
//            const char ch[] = {char('A' + qrook[BLACK]), '\0'};
//            str::pdebug("qrook[BLACK]", ch);
          } else {
            krook[BLACK] = board::_x(ind);
//            const char ch[] = {char('A' + krook[BLACK]), '\0'};
//            str::pdebug("krook[BLACK]", ch);
          }
        }
        break;
      }
      ++board_index;
    }
    assert(board_index == board::SIZE);
    assert(f.board.length() <= board::SIZE);
    // skip space
    while(isspace(s[i]))++i;
    // active plaer
    assert(index("wWbB", s[i]) != nullptr);
    if(s[i] == 'b' || s[i] == 'B')f.active_player=BLACK;
    ++i;
    // skip space
    while(isspace(s[i]))++i;
    // castlings
    while(!isspace(s[i])) {
      assert(index("KkQqabcdefghABCDEFGH-", s[i]) != nullptr);
      char c = s[i];
      switch(s[i]) {
        case 'K':
          c='A'+krook[WHITE];
//          {
//            const char ch[2] = {char(c), '\0'};
//            str::pdebug("convert K", ch);
//          }
          assert(krook[WHITE] != board::nopos);
        break;
        case 'Q':
          c='A'+qrook[WHITE];
//          {
//            const char ch[2] = {char(c), '\0'};
//            str::pdebug("convert Q", ch);
//          }
          assert(qrook[WHITE] != board::nopos);
        break;
        case 'k':
          c='a'+krook[BLACK];
//          {
//            const char ch[2] = {char(c), '\0'};
//            str::pdebug("convert k", ch);
//          }
          assert(krook[BLACK] != board::nopos);
        break;
        case 'q':
          c='a'+qrook[BLACK];
//          {
//            const char ch[2] = {char(c), '\0'};
//            str::pdebug("convert q", ch);
//          }
          assert(qrook[BLACK] != board::nopos);
        break;
        case '-':break;
        default:;break;
      }
      if(isupper(c)) {
        const pos_t castlfile = c - 'A';
        assert(castlfile == qrook[WHITE] || castlfile == krook[WHITE]);
        if(kingpos[WHITE] != E || (castlfile != A && castlfile != H)) {
          f.traditional = false;
        }
        f.castlings |= bitmask::_pos_pair(1u << (c - 'A'), 0x00);
      } else if(c != '-') {
        const pos_t castlfile = c - 'a';
        assert(castlfile == qrook[BLACK] || castlfile == krook[BLACK]);
        if(kingpos[BLACK] != E || (castlfile != A && castlfile != H)) {
          f.traditional = false;
        }
        f.castlings |= bitmask::_pos_pair(0x00, 1u << (c - 'a'));
      }
      ++i;
    }
//    str::pdebug("traditional", f.traditional);
    // skip space
    while(isspace(s[i]))++i;
    // enpassant
    char probe = s[i];
    if(probe != '-') {
      char x = probe;
      assert(index("abcdefgh", x)!=nullptr);
      x -= 'a';
      char y = s[++i];
      assert(index("12345678", y)!=nullptr);
      y -= '1';
      f.enpassant = board::_pos(A+x, 1+y);
    } else {
      f.enpassant = board::nopos;
      ++i;
    }
    // skip space
    while(isspace(s[i]))++i;
    // half-moves, fullmoves
    if(s[i] == '\0') {
      f.halfmove_clock = 0;
      f.fullmove = 1;
      return f;
    }
    int sc;
    sc = sscanf(&s[i], "%hhu %hu", &f.halfmove_clock, &f.fullmove);
    if(f.fullmove == 0)++f.fullmove;
    assert(sc != -1);
    i += sc;
    // done
    return f;
  }

  FEN load_from_file(std::string fname) {
    FILE *fp = fopen(fname.c_str(), "r");
    assert(fp != nullptr);
    std::string s; char c;
    while((c=fgetc(fp))!=EOF)s+=c;
    fclose(fp);
    return load_from_string(s);;
  }

  const FEN starting_pos = fen::load_from_string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"s);
  const FEN pins_pos = fen::load_from_string("rnb1k1nr/ppppbppp/8/4Q3/4P2q/8/PPPP1PPP/RNB1KBNR w KQkq - 1 4"s);
  const FEN castling_pos = fen::load_from_string("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"s);
  const FEN doublecheck_test_pos = fen::load_from_string("rnbqkbnr/pppp1ppp/8/8/4N3/5N2/PPPPQPPP/R1B1KB1R w KQkq - 8 8"s);
  const FEN check_test_pos = fen::load_from_string("rnbqkb1r/pppp1ppp/5n2/4N3/8/8/PPPPQPPP/RNB1KB1R w KQkq - 2 5"s);
  const FEN promotion_test_pos = fen::load_from_string("1k3n1n/4PPP1/8/8/8/8/1pp1PPPP/4K3 w - - 0 1"s);
  const FEN search_explosion_pos = fen::load_from_string("q2k2q1/2nqn2b/1n1P1n1b/2rnr2Q/1NQ1QN1Q/3Q3B/2RQR2B/Q2K2Q1 w - -"s);

  FEN export_from_board(const Board &board);

  std::string export_as_string(const fen::FEN &f) {
    std::string s = ""s;
    pos_t kingpos[2] = {board::nopos, board::nopos};
    for(pos_t y = 0; y < board::LEN; ++y) {
      pos_t emptycount = 0;
      for(pos_t x = 0; x < board::LEN; ++x) {
        const pos_t ind = board::_pos(A+x, 1+y);
        if(f.board[ind] == ' ') {
          ++emptycount;
          continue;
        }
        if(emptycount > 0) {
          s += std::to_string(emptycount);
          emptycount = 0;
        }
        s += f.board[ind];
        if(f.board[ind] == 'K') {
          kingpos[WHITE] = ind;
        } else if(f.board[ind] == 'k') {
          kingpos[BLACK] = ind;
        }
      }
      if(emptycount > 0) {
        s += std::to_string(emptycount);
        emptycount = 0;
      }
      if(y != board::LEN - 1)s+="/";
    }
    if(f.crazyhouse) {
      s += "/"s + f.subs;
    }
    s += ' ';
    s += (f.active_player == WHITE) ? 'w' : 'b';
    s += ' ';
    if(!f.castlings) {
      s += '-';
    } else {
      for(COLOR color : {WHITE, BLACK}) {
        const pos_t mask = (color == WHITE) ? bitmask::first(f.castlings) : bitmask::second(f.castlings);
        for(pos_t _c = 0; _c < board::LEN; ++_c) {
          const pos_t c = board::LEN - _c - 1;
          if(mask & (1 << c)) {
            char ch = ((color == WHITE) ? 'A' : 'a') + c;
            if(f.traditional) {
              switch(ch) {
                case 'H':ch='K';break;
                case 'A':ch='Q';break;
                case 'h':ch='k';break;
                case 'a':ch='q';break;
                default:break;
              }
            }
            s += ch;
          }
        }
      }
    }
    s += ' ';
    s += (f.enpassant == board::nopos) ? "-"s : board::_pos_str(f.enpassant);
    s += " "s + std::to_string(f.halfmove_clock) + " "s + std::to_string(f.fullmove);
    return s;
  }

  std::string export_as_string(const Board &board) {
    std::string s = export_as_string(export_from_board(board));
    return s;
  }
} // namespace fen
