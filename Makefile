DBGFLAGS = -g3
OPTFLAGS = -Ofast -DNDEBUG -fwhole-program
CXXFLAGS = -std=c++17 -I. -fopt-info -Wall -Wextra -Wno-unused -m64
LDFLAGS =
HPPFILES = $(wildcard *.hpp)

all : dummy_chess dummy_chess_curses dummy_chess_bench_board

dummy_chess: main.cpp $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) main.cpp $(LDFLAGS) -o dummy_chess

dummy_chess_curses: ui.cpp $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) $(shell pkgconf --libs ncurses ncursesw) ui.cpp $(LDFLAGS) $(shell pkgconf --libs ncurses ncursesw) $(LDFLAGS) -o dummy_chess_curses

dummy_chess_bench_board: bench_board.cpp $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) bench_board.cpp $(LDFLAGS) -o dummy_chess_bench_board

clean:
	rm -vf dummy_chess dummy_chess_curses dummy_chess_bench_board
