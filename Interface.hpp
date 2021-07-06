#pragma once


#include <ncurses.h>

#include <Board.hpp>
#include <Engine.hpp>
#include <PGN.hpp>


struct Interface {
  Engine &board;
  pgn::PGN pgn;
  bool shouldClose = false;
  int CELL_MW = 2,
      CELL_MH = 1,
      CELL_PMW = 1,
      CELL_PMH = 0;

  Interface(Engine &board):
    board(board), pgn(board)
  {}

  typedef enum {
    NC_COLOR_NORMAL=1,
    NC_COLOR_WHITE_CELL, NC_COLOR_BLACK_CELL,
    NC_COLOR_CHECK,
    NC_COLOR_SELECTION, NC_COLOR_SELECTED,
    NC_COLOR_ENPASSANT, NC_COLOR_PINS,
    NC_COLOR_CAN_ATTACK,

    NC_COLOR_PGN_CURPLY,

    NC_NO_COLORS
  } ncurses_color_palette;

  void run() {
    setlocale(LC_ALL, "");
    initscr();
    start_color();
    // label, fg, bg
    init_pair(NC_COLOR_NORMAL,      COLOR_WHITE, COLOR_BLACK );
    init_pair(NC_COLOR_WHITE_CELL,  COLOR_BLACK, COLOR_WHITE );
    init_pair(NC_COLOR_BLACK_CELL,  COLOR_WHITE, COLOR_BLACK );
    init_pair(NC_COLOR_CHECK,       COLOR_RED,   COLOR_BLACK );
    init_pair(NC_COLOR_SELECTION,   COLOR_WHITE, COLOR_CYAN  );
    init_pair(NC_COLOR_SELECTED,    COLOR_WHITE, COLOR_BLUE  );
    init_pair(NC_COLOR_ENPASSANT,   COLOR_WHITE, COLOR_GREEN);
    init_pair(NC_COLOR_PINS,        COLOR_WHITE, COLOR_RED);
    init_pair(NC_COLOR_CAN_ATTACK,  COLOR_WHITE, COLOR_YELLOW);

    init_pair(NC_COLOR_PGN_CURPLY,  COLOR_WHITE, COLOR_MAGENTA);

    cbreak();
    noecho(); // do not show typed characters
    keypad(stdscr, true); // attack keyboard to stdscr

    refresh();
    box(stdscr, 0, 0); // create a box in stdscr
    refresh();

    // cursor, basically, but called buf because it keeps the value we want to set
    display();
    while(!shouldClose) {
      keyboard(getch());
      display();
    }
  }

  std::string activePlayer() const {
    return (board.activePlayer() == WHITE) ? "WHITE":"BLACK";
  }

  void nc_set_cell_color(COLOR c) {
    attron(COLOR_PAIR(c==WHITE ? NC_COLOR_WHITE_CELL : NC_COLOR_BLACK_CELL));
  }

  void nc_color_condition(COLOR c, int x, int y) {
    const piece_bitboard_t pins = board.state.pins[board.activePlayer()];
    if(x==sel_x&&y==sel_y){
      attron(COLOR_PAIR(NC_COLOR_SELECTED));
      return;
    } else if(x==cursor_x&&y==cursor_y) {
      attron(COLOR_PAIR(NC_COLOR_SELECTION));
      return;
    } else if(board.enpassant_trace() != board::nopos && board.enpassant_trace() == board::_pos(A+x,1+y)) {
      attron(COLOR_PAIR(NC_COLOR_ENPASSANT));
      return;
    }
    if((cursor_x!=-1&&cursor_y!=-1) || (sel_x!=-1&&sel_y!=-1)) {
      pos_t piece_pos = 0;
      if(sel_x!=-1&&sel_y!=-1)piece_pos = board::_pos(A+sel_x, 1+sel_y);
      else piece_pos = board::_pos(A+cursor_x, 1+cursor_y);
      pos_t hit_pos = board::_pos(A+x, 1+y);
      // single-attacks
      //const piece_bitboard_t attacks = board.state.attacks[piece_pos];
      // single-moves
      const piece_bitboard_t attacks = (board.color_at_pos(piece_pos) == board.activePlayer()) ? board.state.moves[piece_pos] : 0x00;
      //const piece_bitboard_t attacks = board.get_capture_moves_from(piece_pos);
      // multi-attacks
      //const auto &piece = board[piece_pos];
      //piece_bitboard_t attacks = piece.get_attacks(board.get_piece_positions(piece.color), board.get_piece_positions(enemy_of(piece.color)));
      // pin-line
      //const piece_bitboard_t attacks = board.get_pin_line_of(piece_pos);
      if(attacks & (1ULL << hit_pos)) {
        attron(COLOR_PAIR(NC_COLOR_CAN_ATTACK));
        return;
      }
    }
    const COLOR active = board.activePlayer();
//    const piece_bitboard_t highlight = board.get_piece(KING,active).get_moves(board.get_king_pos(active),
//                                                                              board.get_piece_positions(active),
//                                                                              board.get_piece_positions(enemy_of(active)),
//                                                                              board.get_attack_mask(enemy_of(active)),
//                                                                              board.get_castlings_mask());
    //const piece_bitboard_t highlight = board.get_attack_mask(enemy_of(active));
    const piece_bitboard_t highlight = pins;
    //const piece_bitboard_t highlight = board.state_checkline[active];
    //const piece_bitboard_t highlight = board.get_attacks_to(board.get_king_pos(active), enemy_of(active));
    if(highlight & (1ULL << board::_pos(A+x,1+y))) {
      attron(COLOR_PAIR(NC_COLOR_PINS));
      return;
    }
    nc_set_cell_color(c);
  }

  void nc_reset_color() {
    attron(COLOR_PAIR(NC_COLOR_NORMAL));
  }

  void nc_draw_cell_margin_mid(COLOR cell_color, COLOR piece_color, int x, int y) {
    if(piece_color==NEUTRAL)piece_color=cell_color;
    nc_draw_cell_margin_side(cell_color,x,y);
    nc_color_condition(piece_color,x,y);
    for(int c=0;c<CELL_PMW*2+1;++c)addch(' ');
    nc_reset_color();
    nc_draw_cell_margin_side(cell_color,x,y);
  }

  void nc_draw_cell_margin_side(COLOR c, int x, int y) {
    nc_color_condition(c,x,y);
    for(int c=0;c<CELL_MW;++c)addch(' ');
    nc_reset_color();
  }

  void nc_draw_cell_margin_both(COLOR c, int x, int y) {
    nc_draw_cell_margin_mid(c,c,x,y);
  }

  void nc_draw_cell_piece_unicode(const Piece &p, int x, int y) {
    nc_set_cell_color(p.color);
    for(int c=0;c<CELL_PMW;++c)addch(' ');
    if(p.value == EMPTY) {
      addch(' ');
    } else {
      //if(p.value==PAWN  &&p.color==WHITE)addstr("♙");
      //if(p.value==KNIGHT&&p.color==WHITE)addstr("♘");
      //if(p.value==BISHOP&&p.color==WHITE)addstr("♗");
      //if(p.value==ROOK  &&p.color==WHITE)addstr("♖");
      //if(p.value==QUEEN &&p.color==WHITE)addstr("♕");
      //if(p.value==KING  &&p.color==WHITE)addstr("♔");
      //if(p.value==PAWN  &&p.color==BLACK)addstr("♟");
      //if(p.value==KNIGHT&&p.color==BLACK)addstr("♞");
      //if(p.value==BISHOP&&p.color==BLACK)addstr("♝");
      //if(p.value==ROOK  &&p.color==BLACK)addstr("♜");
      //if(p.value==QUEEN &&p.color==BLACK)addstr("♛");
      //if(p.value==KING  &&p.color==BLACK)addstr("♚");
      addch(toupper(p.str()));
    }
    for(int c=0;c<CELL_PMW;++c)addch(' ');
    nc_reset_color();
  }

  COLOR nc_get_cell_color(int x, int y) {
    return (x+y)&1?WHITE:BLACK;
  }

  void nc_draw_board_row_margin(int y, int &top, const int LEFT) {
    for(int mh = 0; mh < CELL_MH; ++mh) {
      for(int x = 0; x < board::LEN; ++x) {
        addch(ACS_VLINE);
        nc_draw_cell_margin_both(nc_get_cell_color(x,y),x,y);
      }
      addch(ACS_VLINE);
      move(top++, LEFT);
    }
  }

  void nc_draw_board_row_piece_margin(int y, int &top, const int LEFT) {
    for(int mh = 0; mh < CELL_PMH; ++mh) {
      for(int x = 0; x < board::LEN; ++x) {
        addch(ACS_VLINE);
        auto piece = board[board::_pos(A+x, 1+y)];
        COLOR cell_color = (x+y)&1?WHITE:BLACK;
        nc_draw_cell_margin_mid(cell_color,piece.color,x,y);
      }
      addch(ACS_VLINE);
      move(top++, LEFT);
    }
  }

  void nc_draw_board_row_piece(int y, int &top, const int LEFT) {
    for(int x = 0; x < board::LEN; ++x) {
      addch(ACS_VLINE);
      if(board[board::_pos(A+x, 1+y)].value != EMPTY) {
        nc_draw_cell_margin_side((x + y) & 1 ? WHITE : BLACK,x,y);
        nc_draw_cell_piece_unicode(board[board::_pos(A+x, 1+y)],x,y);
        nc_draw_cell_margin_side((x + y) & 1 ? WHITE : BLACK,x,y);
      } else {
        nc_draw_cell_margin_both((x + y) & 1 ? WHITE : BLACK,x,y);
      }
    }
    addch(ACS_VLINE);
    move(top++, LEFT);
  }

  typedef enum { ROW_SEP_TOP, ROW_SEP_MID, ROW_SEP_BOTTOM } ROW_SEP;
  void nc_draw_row_sep(ROW_SEP rowsep) {
    const int len = ((CELL_MW+CELL_PMW)*2+1 + 1)*8 + 1; // character length of a cell in the user interface
    for(int i=0;i<len;++i) {
      switch(rowsep) {
        case ROW_SEP_TOP:
          if(i == 0) addch(ACS_ULCORNER);
          else if(i == len - 1)addch(ACS_URCORNER);
          else if(i%((CELL_MW+CELL_PMW)*2+2)==0) addch(ACS_TTEE);
          else addch(ACS_HLINE);
        break;
        case ROW_SEP_MID:
          if(i == 0) addch(ACS_LTEE);
          else if(i == len - 1)addch(ACS_RTEE);
          else if(i%((CELL_MW+CELL_PMW)*2+2)==0) addch(ACS_PLUS);
          else addch(ACS_HLINE);
        break;
        case ROW_SEP_BOTTOM:
          if(i == 0) addch(ACS_LLCORNER);
          else if(i == len - 1)addch(ACS_LRCORNER);
          else if(i%((CELL_MW+CELL_PMW)*2+2)==0) addch(ACS_BTEE);
          else addch(ACS_HLINE);
        break;
      }
    }
  }

  void draw_board(const int TOP, const int LEFT, int &top, int &right) {
    const int len = ((CELL_MW+CELL_PMW)*2+1 + 1)*8 + 1; // character length of a cell in the user interface
    right = LEFT + len;
    attron(COLOR_PAIR(NC_COLOR_NORMAL));
    move(top++, LEFT);
    nc_draw_row_sep(ROW_SEP_TOP);
    for(pos_t y_ = 0; y_ < board::LEN; ++y_) {
      pos_t y = board::LEN - y_ - 1;
      move(top++, LEFT);
      nc_draw_board_row_margin(y, top, LEFT);
      nc_draw_board_row_piece_margin(y, top, LEFT);
      nc_draw_board_row_piece(y, top, LEFT);
      nc_draw_board_row_piece_margin(y, top, LEFT);
      nc_draw_board_row_margin(y, top, LEFT);
      if(y_+1!=board::LEN)nc_draw_row_sep(ROW_SEP_MID);
    }
    nc_draw_row_sep(ROW_SEP_BOTTOM);
  }

  void draw_statusbar(const int LEFT, int &top) {
    move(top + 2, LEFT);
    attron(A_BOLD);
    const COLOR c = board.activePlayer();
    //set statusbar message
    //int len = printw("[ %s ]", activePlayer().c_str());
    //int len = printw("[ %s %hhu %hhu->%hhu ]", activePlayer().c_str(), fen::compress_castlings(board.get_castlings_mask()), from, to);
    int len = printw("[ %s [%hu %hu %hu %hu] %hu", activePlayer().c_str(), board.is_castling(WHITE, QUEEN_SIDE), board.is_castling(WHITE, KING_SIDE),
                                                   board.is_castling(BLACK, QUEEN_SIDE), board.is_castling(BLACK, KING_SIDE), board.enpassant_pawn());
    //int len = printw("[ %s %llx ]", activePlayer().c_str(), board.state_checkline);
    //int len = printw("[ %s %llx %llx %hhu ]", activePlayer().c_str(), board.state_checkline[c], board.state_checkline[enemy_of(c)], board.halfmoves_);
    nc_reset_color();
    for(int i = 0; i < 20 - len; ++i)addch(' ');
  }

  void draw_pgn(const int LEFT, const int TOP, int &top) {
    move(top, LEFT);
    printw("Evaluation: %.5f", board.evaluation == DBL_MAX ? 0 : board.evaluation);
    move(++top, LEFT);
    int turn = 1;
    constexpr size_t initial_margin = 5,
                     ply_length = 9, // ' dxe8=Q++ '
                     space_between = 3;
    constexpr size_t turn_length = initial_margin + ply_length + space_between + ply_length;
    constexpr size_t start_white = initial_margin,
                     start_black = initial_margin + ply_length + space_between;
    for(size_t i = (pgn.size() > 20) ? pgn.size() - 20 : 0; i < pgn.size(); ++i) {
      if(!(i & 1)) {
        move(top, LEFT);
        printw("%d.", turn);
      }
      move(top, !(i & 1) ? LEFT + start_white : LEFT + start_black);
      if(i + 1 == pgn.cur_ply) {
        attron(COLOR_PAIR(NC_COLOR_PGN_CURPLY));
      } else {
        nc_set_cell_color(!(i & 1) ? WHITE : BLACK);
      }
      std::string s = " "s + pgn.ply[i] + " "s;
      while(s.length()<ply_length)s+=' ';
      addstr(s.c_str());
      nc_reset_color();
      for(size_t c=0;c<ply_length+space_between;++c)addch(' ');
      if(i & 1) {
        ++top, ++turn;
      }
    }
    if(!(pgn.size() & 1)) {
      move(top,LEFT);
      for(size_t c=0;c<turn_length;++c)addch(' ');
    }
    if(pgn.ending.length() > 0) {
      move(top, LEFT + turn_length / 2 - pgn.ending.length() / 2);
      addstr(pgn.ending.c_str());
    }
  }

  void display() {
    bkgd(COLOR_PAIR(NC_COLOR_NORMAL)); // changes the background
    int cols, rows;
    getmaxyx(stdscr, rows, cols); // acquires screen height and width
    const int TOP = 4, LEFT = cols * 0.03;

    move(2, LEFT); // move sets the position of the cursor (where printw and addch will write)
    printw("Press ESC to leave.");
    move(3, LEFT);
    printw("Use arrows to navigate.");

    int top = TOP; // y coordinate of where to start writing
    int right = LEFT;
    draw_board(TOP, LEFT, top, right);
    draw_statusbar(LEFT, top);
    top = TOP;
    draw_pgn(right + 5, TOP, top);
    refresh();
  }

  int cursor_x = -1, cursor_y = -1;
  int sel_x = -1, sel_y = -1;
  void keyboard(int ch) {
    switch(ch) {
      case 27:
        shouldClose = true;
      break;
      case KEY_LEFT: case 'h':
        if(0 < cursor_x) {
          --cursor_x;
          if(cursor_y==-1)cursor_y=0;
        } else cursor_x=-1,cursor_y=-1,sel_x=-1;
      break;
      case KEY_RIGHT: case 'l':
        if(board::LEN - 1 > cursor_x) {
          ++cursor_x;
          if(cursor_y==-1)cursor_y=0;
        } else cursor_x=-1,cursor_y=-1,sel_x=-1;
      break;
      case KEY_DOWN: case 'j':
        if(0 < cursor_y) {
          --cursor_y;
          if(cursor_x==-1)cursor_x=0;
        } else cursor_x=-1,cursor_y=-1,sel_y=-1;
      break;
      case KEY_UP: case 'k':
        if(board::LEN - 1 > cursor_y) {
          ++cursor_y;
          if(cursor_x==-1)cursor_x=0;
        } else cursor_x=-1,cursor_y=-1,sel_y=-1;
      break;
      case 10:
        if(sel_x==-1||sel_y==-1)sel_x=cursor_x,sel_y=cursor_y;
        else {
          const pos_t pos_from = board::_pos(A+sel_x, 1+sel_y);
          const pos_t pos_to = board::_pos(A+cursor_x, 1+cursor_y);
          const piece_bitboard_t moves = (board[pos_from].color == board.activePlayer() ? board.state.moves[pos_from] : 0x00);
          if((1ULL << pos_to) & moves && board[pos_from].color == board.activePlayer()) {
            pgn.handle_move(pos_from, pos_to | board::PROMOTE_QUEEN);
            sel_x=-1,sel_y=-1;
          } else {
            sel_x=cursor_x,sel_y=cursor_y;
          }
        }
      break;
      case 'r':
        {
          move_t m = board::nomove;
          if(sel_x==-1||sel_y==-1)m=board.get_random_move();
          else m=board.get_random_move_from(board::_pos(A+sel_x, 1+sel_y));
          if(m != board::nomove) {
            pgn.handle_move(m);
          }
        }
      break;
      case 'e':
        board.evaluation = board.evaluate();
        break;
      case ' ':
        {
          move_t m = board.get_fixed_depth_move(6);
          if(m != board::nomove) {
            pgn.handle_move(m);
          }
        }
      break;
      case KEY_BACKSPACE:
        pgn.retract_move();
      break;
    }
  }

  void close() {
    endwin();
  }
};
