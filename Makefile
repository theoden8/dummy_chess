DBGFLAGS = -g3
OPTFLAGS = -Ofast -DNDEBUG -flto -fwhole-program -funroll-loops -fno-trapping-math -m64 -march=native -DUSE_INTRIN -fno-exceptions
OPTFLAGS_PG = $(OPTFLAGS) -fprofile-generate
OPTFLAGS_PGO = $(OPTFLAGS) -fprofile-use
OPTFLAGS_PGO_MT = $(OPTFLAGS_PGO) -fprofile-correction
PROFFLAGS = -O1 -DNDEBUG -flto -DUSE_INTRIN -pg
CXXFLAGS = -std=c++20 -I. -Wall -Wextra -Wno-unused -Wno-parentheses
# clang:
#CXX = clang++
#CXXFLAGS += -Wno-unused-parameter -Wno-range-loop-construct -Wno-unknown-attributes -Wno-ignored-optimization-argument
# CXXFLAGS += -fopt-info
LDFLAGS = -pthread
HPPFILES = $(wildcard *.hpp) m42.h
NC_CFLAGS =  $(shell pkgconf --cflags ncursesw)
NC_LDFLAGS = $(shell pkgconf --libs ncursesw)
SOURCES = m42.cpp

CORES = $(shell getconf _NPROCESSORS_ONLN)
all :; @$(MAKE) _all -j2
_all : dummy_chess dummy_chess_opt dummy_chess_curses dummy_chess_bench dummy_chess_alphabeta dummy_chess_uci dummy_chess_uci_dbg

m42.cpp:
	$ ./scripts/m42_download

m42.h:
	$ ./scripts/m42_download

dummy_chess: simple.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) simple.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_opt: simple.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) simple.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_curses: ui.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(NC_CFLAGS) ui.cpp $(SOURCES) $(LDFLAGS) $(NC_LDFLAGS) $(LDFLAGS) -o $@

dummy_chess_bench: bench.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@
	./dummy_chess_bench 10 '1r4k1/1r3pp1/3b3p/3p1qnP/Q1pP3R/2P2PP1/PP4K1/R1B3N1 b - - 2 24'
	$(CXX) $(OPTFLAGS_PGO) $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_alphabeta: alphabeta.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(PROFFLAGS) -fprofile-generate $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@
	./dummy_chess_alphabeta 10 '1r4k1/1r3pp1/3b3p/3p1qnP/Q1pP3R/2P2PP1/PP4K1/R1B3N1 b - - 2 24'
	$(CXX) $(PROFFLAGS) -fprofile-use $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: uci.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS_PG) $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@
	./scripts/pgo_bench.py "./dummy_chess_uci"
	$(CXX) $(OPTFLAGS_PGO_MT) $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci_dbg: uci.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@

clean:
	rm -vf *.o *.gcda
	rm -vf dummy_chess dummy_chess_opt dummy_chess_curses dummy_chess_bench dummy_chess_alphabeta dummy_chess_uci dummy_chess_uci_dbg
