#pragma once


#include <cctype>
#include <strings.h>
#include <string>

#include <Constants.hpp>
#include <Event.hpp>


using namespace std::string_literals;


namespace fen {
  typedef struct _FEN {
    std::string board;
    COLOR active_player;
    pos_t castling_compressed;
    pos_t enpassant;
    pos_t halfmove_clock;
    pos_t fullmove;
  } FEN;

  FEN load_from_string(std::string s) {
    size_t i = 0;
    FEN f = {
      .board = std::string(),
      .active_player = WHITE,
      .castling_compressed = 0x00,
      .enpassant = event::enpassantnotrace,
      .halfmove_clock = 0,
      .fullmove = 0,
    };
    // board
    for(const char *c = s.c_str(); *c != '\0' && *c != ' '; ++c, ++i) {
      if(*c == '/')continue;
      if(isdigit(*c)) {
        for(int j=0;j<*c-'0';++j)f.board+=' ';
      } else {
        f.board += *c;
      }
    }
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
    piece_bitboard_t castlings = 0x00;
    while(!isspace(s[i])) {
      assert(index("KkQq-", s[i]) != nullptr);
      pos_t blackline = board::SIZE-board::LEN;
      switch(s[i]) {
        case 'K': castlings|=0x40ULL; break;
        case 'Q': castlings|=0x04ULL; break;
        case 'k': castlings|=0x40ULL<<blackline; break;
        case 'q': castlings|=0x04ULL<<blackline; break;
        case '-':break;
      }
      ++i;
    }
    f.castling_compressed = event::compress_castlings(castlings);
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
    } else f.enpassant = event::enpassantnotrace;
    // skip space
    while(isspace(s[i]))++i;
    // half-moves
    int sc;
    sc = sscanf(&s[i], "%hhu", &f.halfmove_clock);
    assert(sc != -1);
    i += sc;
    // skip space
    while(isspace(s[i]))++i;
    // fullmoves
    sc = sscanf(&s[i], "%hhu", &f.halfmove_clock);
    assert(sc != -1);
    i += sc;
    // done
    return f;
  }

  FEN load_from_file(std::string fname) {
    FILE *fp = fopen(fname.c_str(), "r");
    assert(fp != nullptr);
    std::string s;
    char c;
    while((c=fgetc(fp))!=EOF)s+=c;
    fclose(fp);
    return load_from_string(s);;
  }

  const FEN starting_pos = fen::load_from_string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"s);
  const FEN castling_pos = fen::load_from_string("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"s);
  const FEN doublecheck_test_pos = fen::load_from_string("rnbqkbnr/pppp1ppp/8/8/4N3/5N2/PPPPQPPP/R1B1KB1R w KQkq - 8 8"s);
  const FEN check_test_pos = fen::load_from_string("rnbqkb1r/pppp1ppp/5n2/4N3/8/8/PPPPQPPP/RNB1KB1R w KQkq - 2 5"s);
  const FEN promotion_test_pos = fen::load_from_string("1k3n1n/4PPP1/8/8/8/8/1pp1PPPP/4K3 w - - 0 1"s);
} // namespace fen
