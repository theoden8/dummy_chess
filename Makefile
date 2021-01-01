OPTFLAGS = -g3
CXXFLAGS = $(OPTFLAGS) -std=c++17 -I. -fopt-info -Wall -Wextra -Wno-unused -m64
LDFLAGS =
HPPFILES = $(wildcard *.hpp)

all : dummy_chess dummy_chess_curses

dummy_chess: main.cpp $(HPPFILES) Makefile
	$(CXX) $(CXXFLAGS) main.cpp $(LDFLAGS) -o dummy_chess

dummy_chess_curses: ui.cpp $(HPPFILES) Makefile
	$(CXX) $(CXXFLAGS) $(shell pkgconf --libs ncurses ncursesw) ui.cpp $(LDFLAGS) $(shell pkgconf --libs ncurses ncursesw) $(LDFLAGS) -o dummy_chess_curses

clean:
	rm -vf dummy_chess dummy_chess_curses
