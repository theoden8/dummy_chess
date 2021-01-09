DBGFLAGS = -g3
OPTFLAGS = -Ofast -DNDEBUG -flto -fwhole-program -march=native
CXXFLAGS = -std=c++17 -I. -Wall -Wextra -Wno-unused -Wno-parentheses -m64
# CXXFLAGS += -fopt-info
LDFLAGS =
HPPFILES = $(wildcard *.hpp)
NC_CFLAGS =  $(shell pkgconf --libs ncurses ncursesw)
NC_LDFLAGS = $(shell pkgconf --libs ncurses ncursesw)
SOURCES = m42.cpp

CORES = $(shell getconf _NPROCESSORS_ONLN)
all :; @$(MAKE) _all -j$(CORES)
_all : dummy_chess dummy_chess_curses dummy_chess_bench_board dummy_chess_perft

dummy_chess: main.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) main.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_curses: ui.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) $(NC_CFLAGS) ui.cpp $(SOURCES) $(LDFLAGS) $(NC_LDFLAGS) $(LDFLAGS) -o $@

dummy_chess_bench_board: bench_board.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) bench_board.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_perft: perft.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) perft.cpp $(SOURCES) $(LDFLAGS) -o $@

clean:
	rm -vf *.o
	rm -vf dummy_chess dummy_chess_curses dummy_chess_bench_board
