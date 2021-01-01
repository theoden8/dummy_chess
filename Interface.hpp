#pragma once

#include <State.hpp>
#include <Board.hpp>

#include <ncurses.h>


struct Interface {
  State &state;
  bool shouldClose = false;
  int CELL_MW = 2,
      CELL_MH = 1,
      CELL_PMW = 1,
      CELL_PMH = 0;

  Interface(State &state):
    state(state)
  {}

  typedef enum {
    COLOR_WHITE_CELL, COLOR_BLACK_CELL,
    COLOR_NORMAL, COLOR_CHECK,
    COLOR_SELECTION, COLOR_SELECTED,
    COLOR_CAN_ATTACK,
  } ncurses_color_palette;

  void run() {
    initscr();
    start_color();
    init_pair(COLOR_NORMAL,      COLOR_WHITE, COLOR_BLACK );
    init_pair(COLOR_WHITE_CELL,  COLOR_WHITE, COLOR_BLACK );
    init_pair(COLOR_BLACK_CELL,  COLOR_BLACK, COLOR_WHITE );
    init_pair(COLOR_CHECK,       COLOR_RED,   COLOR_BLACK );
    init_pair(COLOR_SELECTION,   COLOR_WHITE, COLOR_CYAN  );
    init_pair(COLOR_SELECTED,    COLOR_WHITE, COLOR_BLUE  );
    init_pair(COLOR_CAN_ATTACK,  COLOR_WHITE, COLOR_YELLOW);

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
    return (state.activePlayer() == WHITE) ? "WHITE":"BLACK";
  }

  void nc_set_cell_color(COLOR c) {
    attron(COLOR_PAIR(c==WHITE ? COLOR_WHITE_CELL : COLOR_BLACK_CELL));
  }

  void nc_color_condition(COLOR c, int x, int y) {
    if(x==sel_x&&y==sel_y){
      attron(COLOR_PAIR(COLOR_SELECTED));
      return;
    } else if(x==cursor_x&&y==cursor_y) {
      attron(COLOR_PAIR(COLOR_SELECTION));
      return;
    }
    if(cursor_x!=-1&&cursor_y!=-1) {
      pos_t piece_pos = Board::_pos(A+cursor_x, 1+cursor_y);
      pos_t hit_pos = Board::_pos(A+x, 1+y);
      piece_bitboard_t attacks = state.get_attacks_from(piece_pos);
      if(attacks & (UINT64_C(1) << (hit_pos))) {
        attron(COLOR_PAIR(COLOR_CAN_ATTACK));
        return;
      }
    }
    nc_set_cell_color(c);
  }

  void nc_reset_color() {
    attron(COLOR_PAIR(COLOR_NORMAL));
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
      addch(tolower(p.str()));
    }
    for(int c=0;c<CELL_PMW;++c)addch(' ');
    nc_reset_color();
  }

  COLOR nc_get_cell_color(int x, int y) {
    return (x+y)&1?WHITE:BLACK;
  }

  void nc_draw_board_row_margin(int y, int &top, const int LEFT) {
    for(int mh = 0; mh < CELL_MH; ++mh) {
      for(int x = 0; x < Board::LENGTH; ++x) {
        addch(ACS_VLINE);
        nc_draw_cell_margin_both(nc_get_cell_color(x,y),x,y);
      }
      addch(ACS_VLINE);
      move(top++, LEFT);
    }
  }

  void nc_draw_board_row_piece_margin(int y, int &top, const int LEFT) {
    for(int mh = 0; mh < CELL_PMH; ++mh) {
      for(int x = 0; x < Board::LENGTH; ++x) {
        addch(ACS_VLINE);
        auto piece = state.at_pos(A+x, 1+y);
        COLOR cell_color = (x+y)&1?WHITE:BLACK;
        nc_draw_cell_margin_mid(cell_color,piece.color,x,y);
      }
      addch(ACS_VLINE);
      move(top++, LEFT);
    }
  }

  void nc_draw_board_row_piece(int y, int &top, const int LEFT) {
    for(int x = 0; x < Board::LENGTH; ++x) {
      addch(ACS_VLINE);
      if(state.at_pos(A+x, 1+y).value != EMPTY) {
        nc_draw_cell_margin_side((x + y) & 1 ? WHITE : BLACK,x,y);
        nc_draw_cell_piece_unicode(state.at_pos(A+x, 1+y),x,y);
        nc_draw_cell_margin_side((x + y) & 1 ? WHITE : BLACK,x,y);
      } else {
        nc_draw_cell_margin_both((x + y) & 1 ? WHITE : BLACK,x,y);
      }
    }
    addch(ACS_VLINE);
    move(top++, LEFT);
  }

  void display() {
    bkgd(COLOR_PAIR(COLOR_NORMAL)); // changes the background
    int cols, rows;
    getmaxyx(stdscr, rows, cols); // acquires screen height and width
    const int TOP = 4, LEFT = cols * 0.03;

    move(2, LEFT); // move sets the position of the cursor (where printw and addch will write)
    printw("Press ESC to leave.");
    move(3, LEFT);
    printw("Use arrows to navigate.");

  print_board:;
    int len = ((CELL_MW+CELL_PMW)*2+1 + 1)*8 + 1; // character length of a cell in the user interface
    attron(COLOR_PAIR(COLOR_NORMAL));
    int top = TOP; // y coordinate of where to start writing
    move(top++, LEFT);
    for(int i=0;i<len;++i) {
      if(i == 0) addch(ACS_ULCORNER);
      else if(i == len - 1)addch(ACS_URCORNER);
      else if(i%((CELL_MW+CELL_PMW)*2+2)==0) addch(ACS_TTEE);
      else addch(ACS_HLINE);
    }
    for(int y = 0; y < Board::LENGTH; ++y) {
      move(top++, LEFT);
      nc_draw_board_row_margin(y, top, LEFT);
      nc_draw_board_row_piece_margin(y, top, LEFT);
      nc_draw_board_row_piece(y, top, LEFT);
      nc_draw_board_row_piece_margin(y, top, LEFT);
      nc_draw_board_row_margin(y, top, LEFT);
      for(int i=0;i<len;++i) {
        if(y < Board::LENGTH - 1) {
          if(i == 0) addch(ACS_LTEE);
          else if(i == len - 1)addch(ACS_RTEE);
          else if(i%((CELL_MW+CELL_PMW)*2+2)==0) addch(ACS_PLUS);
          else addch(ACS_HLINE);
        } else {
          if(i == 0) addch(ACS_LLCORNER);
          else if(i == len - 1)addch(ACS_LRCORNER);
          else if(i%((CELL_MW+CELL_PMW)*2+2)==0) addch(ACS_BTEE);
          else addch(ACS_HLINE);
        }
      }
    }
  print_statusbar:;
    move(top + 2, LEFT);
    attron(A_BOLD);
    //set_statusbar_message
    len = printw("[ %s ]", activePlayer().c_str());
    nc_reset_color();
    for(int i = 0; i < 20 - len; ++i)
      addch(' ');
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
        if(Board::LENGTH - 1 > cursor_x) {
          ++cursor_x;
          if(cursor_y==-1)cursor_y=0;
        } else cursor_x=-1,cursor_y=-1,sel_x=-1;
      break;
      case KEY_UP: case 'k':
        if(0 < cursor_y) {
          --cursor_y;
          if(cursor_x==-1)cursor_x=0;
        } else cursor_x=-1,cursor_y=-1,sel_y=-1;
      break;
      case KEY_DOWN: case 'j':
        if(Board::LENGTH - 1 > cursor_y) {
          ++cursor_y;
          if(cursor_x==-1)cursor_x=0;
        } else cursor_x=-1,cursor_y=-1,sel_y=-1;
      break;
      case 10:
        if(sel_x==cursor_x&&sel_y==cursor_y)sel_x=-1,sel_y=-1;
        else sel_x=cursor_x,sel_y=cursor_y;
      break;
    }
  }

  void close() {
    endwin();
  }
};
