#pragma once

#include <State.hpp>
#include <Board.hpp>

#include <cstring>

#include <ncurses.h>


struct Interface {
  State &state;
  bool shouldClose = false;
  static constexpr int CELL_MW = 3, CELL_MH = 1;

  Interface(State &state):
    state(state)
  {}

  typedef enum {
    COLOR_WHITE_CELL, COLOR_BLACK_CELL,
    COLOR_NORMAL, COLOR_CHECK
  } ncurses_color_palette;
  void run() {
    initscr();
    start_color();
    init_pair(COLOR_NORMAL,      COLOR_WHITE, COLOR_BLACK);
    init_pair(COLOR_WHITE_CELL,  COLOR_WHITE, COLOR_BLACK);
    init_pair(COLOR_BLACK_CELL,  COLOR_BLACK, COLOR_WHITE);
    init_pair(COLOR_CHECK,       COLOR_RED,   COLOR_BLACK);

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
    if(state.activePlayer() == WHITE) {
      return "WHITE";
    }
    return "BLACK";
  }

  void draw_cell_margin_side(COLOR c) {
    attron(COLOR_PAIR(c==WHITE ? COLOR_WHITE_CELL : COLOR_BLACK_CELL));
    for(int c=0;c<CELL_MW;++c)addch(' ');
    attron(COLOR_PAIR(COLOR_NORMAL));
  }

  void draw_cell_margin_both(COLOR c) {
    attron(COLOR_PAIR(c==WHITE ? COLOR_WHITE_CELL : COLOR_BLACK_CELL));
    for(int c=0;c<CELL_MW*2+1;++c)addch(' ');
    attron(COLOR_PAIR(COLOR_NORMAL));
  }

  void draw_cell_piece_unicode(const Piece &p) {
    if(p.value == EMPTY) {
      addch(' ');
      return;
    }
    addch(p.str());
  }

  void display() {
    bkgd(COLOR_PAIR(COLOR_NORMAL)); // changes the background
    int cols, rows;
    getmaxyx(stdscr, rows, cols); // acquires screen height and width
    const int TOP = rows * 0.05 + 2, LEFT = cols * 0.03;

    move(2, LEFT); // move sets the position of the cursor (where printw and addch will write)
    printw("Press ESC to leave.");

  print_board:;
    int len = (CELL_MW*2+1 + 1)*8 + 1; // character length of a cell in the user interface
    attron(COLOR_PAIR(COLOR_NORMAL));
    int top = TOP; // y coordinate of where to start writing
    move(top++, LEFT);
    for(int i=0;i<len;++i) {
      if(i == 0) addch(ACS_ULCORNER);
      else if(i == len - 1)addch(ACS_URCORNER);
      else if(i%(CELL_MW*2+2)==0) addch(ACS_TTEE);
      else addch(ACS_HLINE);
    }
    for(int y = 0; y < Board::LENGTH; ++y) {
      move(top++, LEFT);
      for(int mh = 0; mh < CELL_MH; ++mh) {
        for(int x = 0; x < Board::LENGTH; ++x) {
          addch(ACS_VLINE);
          draw_cell_margin_both((x + y) & 1 ? WHITE : BLACK);
        }
        addch(ACS_VLINE);
        move(top++, LEFT);
      }
      for(int x = 0; x < Board::LENGTH; ++x) {
        addch(ACS_VLINE);
        draw_cell_margin_side((x + y) & 1 ? WHITE : BLACK);
        draw_cell_piece_unicode(state.at_pos(y*Board::LENGTH+x));
        draw_cell_margin_side((x + y) & 1 ? WHITE : BLACK);
      }
      addch(ACS_VLINE);
      move(top++, LEFT);
      for(int mh = 0; mh < CELL_MH; ++mh) {
        for(int x = 0; x < Board::LENGTH; ++x) {
          addch(ACS_VLINE);
          draw_cell_margin_both((x + y) & 1 ? WHITE : BLACK);
        }
        addch(ACS_VLINE);
        move(top++, LEFT);
      }
      for(int i=0;i<len;++i) {
        if(y < Board::LENGTH - 1) {
          if(i == 0) addch(ACS_LTEE);
          else if(i == len - 1)addch(ACS_RTEE);
          else if(i%(CELL_MW*2+2)==0) addch(ACS_PLUS);
          else addch(ACS_HLINE);
        } else {
          if(i == 0) addch(ACS_LLCORNER);
          else if(i == len - 1)addch(ACS_LRCORNER);
          else if(i%(CELL_MW*2+2)==0) addch(ACS_BTEE);
          else addch(ACS_HLINE);
        }
      }
    }
  print_statusbar:;
    move(top + 2, LEFT);
    attron(A_BOLD);
    //set_statusbar_message
    //actually_print_it
    len = printw("[ %s ]", activePlayer().c_str());
    attron(COLOR_PAIR(COLOR_NORMAL));
    for(int i = 0; i < 20 - len; ++i)
      addch(' ');
    refresh();
  }

  int cursor_x = -1, cursor_y = -1;
  void keyboard(int ch) {
    switch(ch) {
      case 27:
        shouldClose = true;
      break;
      case KEY_LEFT: case 'h':
        if(0 < cursor_x) {
          --cursor_x;
        }
      break;
      case KEY_RIGHT: case 'l':
        if(Board::LENGTH - 1 > cursor_x) {
          ++cursor_x;
        }
      break;
      case KEY_UP: case 'k':
        if(0 < cursor_y) {
          --cursor_y;
        }
      break;
      case KEY_DOWN: case 'j':
        if(Board::LENGTH - 1 > cursor_y) {
          ++cursor_y;
        }
      break;
      break;
    }
  }

  void close() {
    endwin();
  }
};
