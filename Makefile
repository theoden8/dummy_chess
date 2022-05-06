.PHONY: all _all clean test

DBGFLAGS := -g3

FEATURE_SUPPORT_SANITIZE = $(shell ./scripts/compiler_support_sanitize $(CXX))
FEATURE_SUPPORT_PGO = $(shell ./scripts/compiler_support_pgo $(CXX))
FEATURE_SUPPORT_GCC = $(shell ./scripts/compiler_support_gccflags $(CXX))
FEATURE_SUPPORT_CLANG = $(shell ./scripts/compiler_support_clangflags $(CXX))

$(info CXX is $(CXX))
$(info FEATURE_SUPPORT_SANITIZE is $(FEATURE_SUPPORT_SANITIZE))
$(info FEATURE_SUPPORT_PGO is $(FEATURE_SUPPORT_PGO))
$(info FEATURE_SUPPORT_GCC is $(FEATURE_SUPPORT_GCC))
$(info FEATURE_SUPPORT_CLANG is $(FEATURE_SUPPORT_CLANG))

# sanitization
ifeq ($(FEATURE_SUPPORT_SANITIZE),enabled)
  ifeq ($(FEATURE_SUPPORT_GCC),gcc)
    DBGFLAGS := -static-libasan $(DBGFLAGS)
  endif
  DBGFLAGS := $(DBGFLAGS) -fsanitize=address -fsanitize=undefined
endif

PROFFLAGS = -O1 -DNDEBUG -flto -DUSE_INTRIN -pg
OPTFLAGS := -Ofast -DNDEBUG -flto -fno-trapping-math -fno-signed-zeros -m64 -march=native -DUSE_INTRIN -fno-exceptions

ifeq ($(FEATURE_SUPPORT_GCC),gcc)
  OPTFLAGS := -fwhole-program $(OPTFLAGS)
endif

CXXFLAGS := -std=c++20 -I. -Wall -Wextra
LDFLAGS := -pthread
ifeq ($(FEATURE_SUPPORT_GCC),gcc)
  CXXFLAGS := $(CXXFLAGS) -Wno-unused -Wno-parentheses
else ifeq ($(FEATURE_SUPPORT_CLANG),clang)
  CXXFLAGS := $(CXXFLAGS) -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
                          -Wno-range-loop-construct -Wno-unknown-attributes -Wno-parentheses
  LDFLAGS := $(LDFLAGS)
endif
# CXXFLAGS += -fopt-info

PKGCONFIG ?= $(shell ./scripts/command_pkgconfig)
LLVM_PROFDATA ?= llvm-profdata
NC_CFLAGS =  $(shell $(PKGCONFIG) --cflags ncursesw)
NC_LDFLAGS = $(shell $(PKGCONFIG) --libs ncursesw)

HPPFILES = $(wildcard *.hpp) m42.h
SOURCES = m42.cpp

CORES = $(shell getconf _NPROCESSORS_ONLN)
all :; @$(MAKE) _all -j$(CORES)
_all : dummy_chess dummy_chess_opt dummy_chess_curses dummy_chess_bench dummy_chess_alphabeta dummy_chess_uci dummy_chess_uci_dbg

m42.cpp:
	./scripts/m42_download

m42.h:
	./scripts/m42_download

dummy_chess: simple.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) simple.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_opt: simple.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) simple.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_curses: ui.cpp $(SOURCES) $(HPPFILES) Makefile
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(NC_CFLAGS) ui.cpp $(SOURCES) $(LDFLAGS) $(NC_LDFLAGS) $(LDFLAGS) -o $@

DEPS_BENCH := bench.cpp $(SOURCES) $(HPPFILES) Makefile dummy_chess_uci
DEPS_ALPHABETA := alphabeta.cpp $(SOURCES) $(HPPFILES) Makefile dummy_chess_uci
DEPS_UCI := uci.cpp $(SOURCES) $(HPPFILES) Makefile
ifeq ($(FEATURE_SUPPORT_PGO),disabled)
dummy_chess_bench: $(DEPS_BENCH)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_alphabeta: $(DEPS_ALPHABETA)
	$(CXX) $(PROFFLAGS) $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@
else ifeq ($(FEATURE_SUPPORT_PGO),gcc)
dummy_chess_bench: $(DEPS_BENCH)
	$(CXX) $(OPTFLAGS) -fprofile-use $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_alphabeta: $(DEPS_ALPHABETA)
	$(CXX) $(PROFFLAGS) -fprofile-use -fprofile-correction $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	rm -vf *.gcda
	$(CXX) $(OPTFLAGS) -fprofile-generate $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@
	./scripts/pgo_bench.py "./dummy_chess_uci"
	$(CXX) $(OPTFLAGS) -fprofile-use -fprofile-correction $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@
else ifeq ($(FEATURE_SUPPORT_PGO),clang)
dummy_chess_bench: $(DEPS_BENCH)
	$(CXX) $(OPTFLAGS) -fprofile-use=uci.profdata $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_alphabeta: $(DEPS_ALPHABETA)
	$(CXX) $(PROFFLAGS) -fprofile-use=uci.profdata $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	rm -rvf uci.d.profdata
	rm -vf uci.profdata
	$(CXX) $(OPTFLAGS) -fprofile-generate=uci.d.profdata $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@
	./scripts/pgo_bench.py "./dummy_chess_uci"
	$(LLVM_PROFDATA) merge -output=uci.profdata uci.d.profdata
	$(CXX) $(OPTFLAGS) -fprofile-use=uci.profdata $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@
endif

dummy_chess_uci_dbg: $(DEPS_UCI)
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@

test:
	./scripts/perft_test.py

clean:
	rm -vf *.o *.gcda uci.profdata
	rm -rvf uci.d.profdata
	rm -vf dummy_chess dummy_chess_opt dummy_chess_curses dummy_chess_bench dummy_chess_alphabeta dummy_chess_uci dummy_chess_uci_dbg
