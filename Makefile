DBGFLAGS = -g3 -DZOBRIST_SIZE="(1<<12)"
OPTFLAGS = -Ofast -DNDEBUG -flto -fwhole-program -march=native
PROFFLAGS = $(OPTFLAGS) -pg
CXXFLAGS = -std=c++20 -I. -Wall -Wextra -Wno-unused -Wno-parentheses -m64
# clang:
#CXX = clang++-13
#CXXFLAGS += -Wno-unused-parameter -Wno-range-loop-construct -Wno-unknown-attributes -Wno-ignored-optimization-argument
# CXXFLAGS += -fopt-info
LDFLAGS =
HPPFILES = $(wildcard *.hpp)
NC_CFLAGS =  $(shell pkgconf --cflags ncursesw)
NC_LDFLAGS = $(shell pkgconf --libs ncursesw)
SOURCES = m42.cpp

CORES = $(shell getconf _NPROCESSORS_ONLN)
all :; @$(MAKE) _all -j$(CORES)
_all : dummy_chess dummy_chess_opt dummy_chess_abwalk dummy_chess_playout dummy_chess_curses dummy_chess_curses_rel dummy_chess_bench dummy_chess_perft dummy_chess_uci

dummy_chess: simple.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) simple.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_opt: simple.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) simple.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_abwalk: abwalk.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) abwalk.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_playout: playout.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) playout.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_curses: ui.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) $(NC_CFLAGS) ui.cpp $(SOURCES) $(LDFLAGS) $(NC_LDFLAGS) $(LDFLAGS) -o $@

dummy_chess_curses_rel: ui.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(NC_CFLAGS) ui.cpp $(SOURCES) $(LDFLAGS) $(NC_LDFLAGS) $(LDFLAGS) -o $@

dummy_chess_bench: bench.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_perft: perft.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(PROFFLAGS) $(CXXFLAGS) perft.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: uci.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@

clean:
	rm -vf *.o
	rm -vf dummy_chess dummy_chess_curses dummy_chess_curses_rel dummy_chess_bench dummy_chess_perft dummy_chess_uci
