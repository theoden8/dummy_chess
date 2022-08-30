.PHONY: all _all clean test

DBGFLAGS := -g3

FEATURE_SUPPORT_SANITIZE ?= $(shell ./scripts/compiler_support_sanitize $(CXX))
FEATURE_SUPPORT_PGO ?= $(shell ./scripts/compiler_support_pgo $(CXX))
FEATURE_SUPPORT_GCC ?= $(shell ./scripts/compiler_support_gccflags $(CXX))
FEATURE_SUPPORT_CLANG ?= $(shell ./scripts/compiler_support_clangflags $(CXX))
FEATURE_SUPPORT_LIBBSD ?= $(shell ./scripts/compiler_support_libbsd $(CXX))
FEATURE_SUPPORT_JEMALLOC ?= $(shell ./scripts/compiler_support_jemalloc $(CXX))
FEATURE_SUPPORT_STDRANGES ?= $(shell ./scripts/compiler_support_stdranges $(CXX))

$(info CXX is $(CXX))
$(info FEATURE_SUPPORT_SANITIZE is $(FEATURE_SUPPORT_SANITIZE))
$(info FEATURE_SUPPORT_PGO is $(FEATURE_SUPPORT_PGO))
$(info FEATURE_SUPPORT_GCC is $(FEATURE_SUPPORT_GCC))
$(info FEATURE_SUPPORT_CLANG is $(FEATURE_SUPPORT_CLANG))
$(info FEATURE_SUPPORT_LIBBSD is $(FEATURE_SUPPORT_LIBBSD))
$(info FEATURE_SUPPORT_JEMALLOC is $(FEATURE_SUPPORT_JEMALLOC))
$(info FEATURE_SUPPORT_STDRANGES is $(FEATURE_SUPPORT_STDRANGES))

# sanitization
ifeq ($(FEATURE_SUPPORT_SANITIZE),enabled)
  ifeq ($(FEATURE_SUPPORT_GCC),gcc)
    DBGFLAGS := -static-libasan $(DBGFLAGS)
  endif
  DBGFLAGS := $(DBGFLAGS) -fsanitize=address -fsanitize=undefined
endif

PROFFLAGS = -O1 -DNDEBUG -flto -DUSE_INTRIN -pg
OPTFLAGS := -Ofast -DNDEBUG -flto -fno-trapping-math -fno-signed-zeros -m64 -march=native -DUSE_INTRIN -fno-exceptions

PKGCONFIG ?= $(shell ./scripts/command_pkgconfig)
CXXFLAGS := -std=c++20 -I. -Wall -Wextra
LDFLAGS := -pthread
# compiler-specific
ifeq ($(FEATURE_SUPPORT_GCC),gcc)
  OPTFLAGS := -fwhole-program $(OPTFLAGS)
  CXXFLAGS := $(CXXFLAGS) -Wno-unused -Wno-parentheses
else ifeq ($(FEATURE_SUPPORT_CLANG),clang)
  CXXFLAGS := $(CXXFLAGS) -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
                          -Wno-range-loop-construct -Wno-unknown-attributes -Wno-parentheses
  LDFLAGS := $(LDFLAGS)
endif

# bsd
ifeq ($(FEATURE_SUPPORT_LIBBSD),disabled)
  CXXFLAGS := $(CXXFLAGS) -DFLAG_BSD
  LDFLAGS := -lbsd $(LDFLAGS)
endif

# jemalloc
ifeq ($(FEATURE_SUPPORT_JEMALLOC),builtin)
  CXXFLAGS := $(CXXFLAGS) -DFLAG_JEMALLOC_BUILTIN
else ifeq ($(FEATURE_SUPPORT_JEMALLOC),external)
  CXXFLAGS := $(CXXFLAGS) $(shell $(PKGCONFIG) --cflags jemalloc) -DFLAG_JEMALLOC_EXTERNAL
  LDFLAGS := $(LDFLAGS) $(shell $(PKGCONFIG) --libs jemalloc)
else ifeq ($(FEATURE_SUPPORT_JEMALLOC),disabled)
  CXXFLAGS := $(CXXFLAGS) -DFLAG_JEMALLOC_DISABLED
endif

# ranges
ifeq ($(FEATURE_SUPPORT_STDRANGES),disabled)
  CXXFLAGS := $(CXXFLAGS) -DFLAG_STDRANGES
endif
# CXXFLAGS += -fopt-info

LLVM_PROFDATA ?= llvm-profdata
NC_CFLAGS =  $(shell $(PKGCONFIG) --cflags ncursesw)
NC_LDFLAGS = $(shell $(PKGCONFIG) --libs ncursesw)

HPPFILES = $(wildcard *.hpp) m42.h tbconfig.h
SOURCE_DEPS = external/syzygy external/fathom
SOURCES = m42.cpp

CORES = $(shell getconf _NPROCESSORS_ONLN)
all :; @$(MAKE) _all -j$(CORES)
_all : dummy_chess dummy_chess_curses dummy_chess_bench dummy_chess_alphabeta dummy_chess_uci dummy_chess_uci_dbg $(SOURCE_DEPS)

external/syzygy:
	./scripts/syzygy_download

external/fathom:
	./scripts/fathom_download

m42.cpp:
	./scripts/m42_download

m42.h:
	./scripts/m42_download

dummy_chess: simple.cpp $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) simple.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_curses: ui.cpp $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(NC_CFLAGS) ui.cpp $(SOURCES) $(LDFLAGS) $(NC_LDFLAGS) $(LDFLAGS) -o $@

DEPS_BENCH := bench.cpp $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
DEPS_ALPHABETA := alphabeta.cpp $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
DEPS_UCI := uci.cpp $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
ifeq ($(FEATURE_SUPPORT_PGO),disabled)
dummy_chess_bench: $(DEPS_BENCH)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_alphabeta: $(DEPS_ALPHABETA)
	$(CXX) $(PROFFLAGS) $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) -DMUTE_ERRORS uci.cpp $(SOURCES) $(LDFLAGS) -o $@
else ifeq ($(FEATURE_SUPPORT_PGO),gcc)
dummy_chess_bench: $(DEPS_BENCH) dummy_chess_uci
	$(CXX) $(OPTFLAGS) -fprofile-use $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_alphabeta: $(DEPS_ALPHABETA) dummy_chess_uci
	$(CXX) $(PROFFLAGS) -fprofile-use -fprofile-correction $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	rm -vf *.gcda
	$(CXX) $(OPTFLAGS) -fprofile-generate $(CXXFLAGS) -DMUTE_ERRORS uci.cpp $(SOURCES) $(LDFLAGS) -o $@
	./scripts/pgo_bench.py "./dummy_chess_uci"
	$(CXX) $(OPTFLAGS) -fprofile-use -fprofile-correction $(CXXFLAGS) -DMUTE_ERRORS uci.cpp $(SOURCES) $(LDFLAGS) -o $@
else ifeq ($(FEATURE_SUPPORT_PGO),clang)
dummy_chess_bench: $(DEPS_BENCH) dummy_chess_uci
	$(CXX) $(OPTFLAGS) -fprofile-use=uci.profdata $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_alphabeta: $(DEPS_ALPHABETA) dummy_chess_uci
	$(CXX) $(PROFFLAGS) -fprofile-use=uci.profdata $(CXXFLAGS) -Wno-backend-plugin alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	rm -rvf uci.d.profdata
	rm -vf uci.profdata
	$(CXX) $(OPTFLAGS) -fprofile-generate=uci.d.profdata -DMUTE_ERRORS $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@
	./scripts/pgo_bench.py "./dummy_chess_uci"
	$(LLVM_PROFDATA) merge -output=uci.profdata uci.d.profdata
	$(CXX) $(OPTFLAGS) -fprofile-use=uci.profdata $(CXXFLAGS) -DMUTE_ERRORS uci.cpp $(SOURCES) $(LDFLAGS) -o $@
endif

dummy_chess_uci_dbg: $(DEPS_UCI)
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@

test:
	./scripts/perft_test.py

clean:
	rm -vf *.o *.gcda uci.profdata
	rm -rvf uci.d.profdata
	rm -vf dummy_chess dummy_chess_opt dummy_chess_curses dummy_chess_bench dummy_chess_alphabeta dummy_chess_uci dummy_chess_uci_dbg
