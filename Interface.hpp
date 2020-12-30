#pragma once

#include <State.hpp>
#include <Board.hpp>

#include <ncurses.h>


struct Interface {
  State &state;
  bool shouldClose = false;

  Interface(State &state):
    state(state)
  {}

  int NORMAL = -2;
  int CHECK = -1;
  void run() {
    int ch;
    initscr();
    start_color();
    init_pair(NORMAL, COLOR_WHITE, COLOR_BLACK);
    init_pair(WHITE,  COLOR_WHITE, COLOR_BLACK);
    init_pair(BLACK,  COLOR_BLACK, COLOR_WHITE);
    init_pair(CHECK,  COLOR_RED, COLOR_BLACK);

    cbreak();
    noecho(); // do not show typed characters
    keypad(stdscr, TRUE); // attack keyboard to stdscr

    refresh();
    box(stdscr, 0, 0); // create a box in stdscr
    refresh();

    // cursor, basically, but called buf because it keeps the value we want to set
    while(!shouldClose) {
      idle();
    }
  }

  void idle() {
    bkgd(COLOR_PAIR(NORMAL)); // changes the background
    int cols, rows;
    getmaxyx(stdscr, rows, cols); // acquires screen height and width
    const int TOP = rows * 0.05 + 2, LEFT = cols * 0.03;

    move(2, LEFT); // move sets the position of the cursor (where printw and addch will write)
    printw("Press ESC to leave.");

  print_board:;
    int len = 4*8 + 1; // character length of a cell in the user interface
    attron(COLOR_PAIR(NORMAL));
    int top = TOP; // y coordinate of where to start writing
    for(int y = 0; y < Board::LENGTH; ++y) {
      move(top, LEFT);
      for(int x = 0; x < Board::LENGTH; ++x) {
        addch(state.at_pos(y*Board::LENGTH+x).str());
      }
      ++top;
      move(top, LEFT);
      ++top;
    }
  print_statusbar:;
    move(top + 2, LEFT);
    attron(A_BOLD);
    //set_statusbar_message
    //actually_print_it
    len = printw("[ %s ]", std::to_string(BLACK).c_str());
    attron(COLOR_PAIR(NORMAL));
    for(int i = 0; i < 20 - len; ++i)
      addch(' ');
    refresh();
  }

  void close() {
  }
};
