.PHONY: all _all clean test

DBGFLAGS := -g3

FEATURE_SUPPORT_SANITIZE ?= $(shell ./scripts/compiler_support_sanitize $(CC))
FEATURE_SUPPORT_PGO ?= $(shell ./scripts/compiler_support_pgo $(CXX))
FEATURE_SUPPORT_GCC ?= $(shell ./scripts/compiler_support_gccflags $(CXX))
FEATURE_SUPPORT_CLANG ?= $(shell ./scripts/compiler_support_clangflags $(CXX))
FEATURE_SUPPORT_LIBBSD ?= $(shell ./scripts/compiler_support_libbsd $(CXX))
FEATURE_SUPPORT_JEMALLOC ?= $(shell ./scripts/compiler_support_jemalloc $(CXX))
FEATURE_SUPPORT_STDRANGES ?= $(shell ./scripts/compiler_support_stdranges $(CXX))
FEATURE_SUPPORT_GPROF ?= $(shell ./scripts/compiler_support_gprof $(CC))
FLAG_THREADS ?= $(shell ./scripts/compiler_flag_threads $(CXX))

$(info CXX is $(CXX))
$(info FEATURE_SUPPORT_SANITIZE is $(FEATURE_SUPPORT_SANITIZE))
$(info FEATURE_SUPPORT_PGO is $(FEATURE_SUPPORT_PGO))
$(info FEATURE_SUPPORT_GCC is $(FEATURE_SUPPORT_GCC))
$(info FEATURE_SUPPORT_CLANG is $(FEATURE_SUPPORT_CLANG))
$(info FEATURE_SUPPORT_LIBBSD is $(FEATURE_SUPPORT_LIBBSD))
$(info FEATURE_SUPPORT_JEMALLOC is $(FEATURE_SUPPORT_JEMALLOC))
$(info FEATURE_SUPPORT_STDRANGES is $(FEATURE_SUPPORT_STDRANGES))
$(info FEATURE_SUPPORT_GPROF is $(FEATURE_SUPPORT_GPROF))
$(info FLAG_THREADS is "$(FLAG_THREADS)")

# sanitization
ifeq ($(FEATURE_SUPPORT_SANITIZE),enabled)
  ifeq ($(FEATURE_SUPPORT_GCC),gcc)
    DBGFLAGS := -static-libasan $(DBGFLAGS)
  endif
  DBGFLAGS := $(DBGFLAGS) -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
else ifeq ($(FEATURE_SUPPORT_SANITIZE),minimal)
  DBGFLAGS := $(DBGFLAGS) -fsanitize-minimal-runtime -fno-omit-frame-pointer
endif

PROFFLAGS = -O1 -DNDEBUG -DFLAG_PROFILING -flto=auto -DUSE_INTRIN -pg
OPTLIBFLAGS := -O3 -ffast-math -DNDEBUG -fno-trapping-math -fno-signed-zeros -march=native -DUSE_INTRIN -fno-exceptions
ifneq ($(filter x86_64 amd64,$(shell uname -m)),)
  OPTLIBFLAGS := $(OPTLIBFLAGS) -m64
endif
OPTFLAGS := $(OPTLIBFLAGS) -flto=auto

PKGCONFIG ?= $(shell ./scripts/command_pkgconfig)
CXXFLAGS := -std=c++20 -I. -Wall -Wextra -fno-stack-protector
LDFLAGS := $(FLAG_THREADS)
# compiler-specific
ifeq ($(FEATURE_SUPPORT_GCC),gcc)
  OPTFLAGS := -fwhole-program $(OPTFLAGS)
  CXXFLAGS := $(CXXFLAGS) -Wno-unused -Wno-parentheses
else ifeq ($(FEATURE_SUPPORT_CLANG),clang)
  CXXFLAGS := $(CXXFLAGS) -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable \
                          -Wno-range-loop-construct -Wno-unknown-attributes -Wno-parentheses
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

NC_CFLAGS =  $(shell $(PKGCONFIG) --cflags ncursesw)
NC_LDFLAGS = $(shell $(PKGCONFIG) --libs ncursesw)

HPPFILES = $(wildcard *.hpp) m42.h tbconfig.h
SOURCE_DEPS = external/syzygy external/fathom
SOURCES = m42.cpp

SHARED_LIB_EXT := so
ifeq ($(shell uname),Darwin)
	SHARED_LIB_EXT := dylib
endif
LIBDUMMYCHESS := libdummychess.$(SHARED_LIB_EXT)
CXXLIBFLAGS := $(CXXFLAGS) -DINLINE= -DFLAG_EXPORT
# Library build type: debug or release (default: release)
OPTION_LIB_BUILD_TYPE ?= release

ifeq ($(OPTION_LIB_BUILD_TYPE),debug)
  LIBBUILDFLAGS := $(DBGFLAGS)
else
  LIBBUILDFLAGS := $(OPTLIBFLAGS)
endif

TARGETS := dummy_chess dummy_chess_curses dummy_chess_bench
ifeq ($(FEATURE_SUPPORT_GPROF),enabled)
	TARGETS := $(TARGETS) dummy_chess_alphabeta
endif
TARGETS += dummy_chess_uci dummy_chess_uci_dbg $(LIBDUMMYCHESS) libdummychess.a

CORES = $(shell getconf _NPROCESSORS_ONLN)
all :; @$(MAKE) _all -j$(CORES)
_all : $(TARGETS) $(SOURCE_DEPS)

external/syzygy:
	MAKE=$(MAKE) ./scripts/syzygy_download

external/fathom:
	MAKE=$(MAKE) ./scripts/fathom_download

m42.cpp:
	./scripts/m42_download

m42.h:
	./scripts/m42_download

dummy_chess: simple.cpp $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) simple.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_curses: ui.cpp $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(NC_CFLAGS) ui.cpp $(SOURCES) $(LDFLAGS) $(NC_LDFLAGS) $(LDFLAGS) -o $@

DEPS_SHARED := shared_object.cpp $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
DEPS_STATIC := $(SOURCES) $(HPPFILES) Makefile $(SOURCE_DEPS)
DEPS_BENCH := bench.cpp $(DEPS_SHARED)
DEPS_ALPHABETA := alphabeta.cpp $(DEPS_SHARED)
DEPS_UCI := uci.cpp $(DEPS_SHARED)

ifeq ($(FEATURE_SUPPORT_PGO),disabled)

dummy_chess_alphabeta: $(DEPS_ALPHABETA)
	$(CXX) $(PROFFLAGS) $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) -DMUTE_ERRORS uci.cpp $(SOURCES) $(LDFLAGS) -o $@

PROFILE_USE :=

else ifeq ($(FEATURE_SUPPORT_PGO),gcc)

dummy_chess_alphabeta: $(DEPS_ALPHABETA) dummy_chess_uci
	$(CXX) $(PROFFLAGS) -fprofile-use -fprofile-correction $(CXXFLAGS) alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	rm -vf *.gcda
	$(CXX) $(OPTFLAGS) -fprofile-generate $(CXXFLAGS) -DMUTE_ERRORS uci.cpp $(SOURCES) $(LDFLAGS) -o $@
	./scripts/pgo_bench.py "./dummy_chess_uci"
	$(CXX) $(OPTFLAGS) -fprofile-use -fprofile-correction $(CXXFLAGS) -DMUTE_ERRORS uci.cpp $(SOURCES) $(LDFLAGS) -o $@

PROFILE_USE := -fprofile-use

else ifeq ($(FEATURE_SUPPORT_PGO),clang)

LLVM_PROFDATA = $(shell $(CXX) -print-prog-name=llvm-profdata)

dummy_chess_alphabeta: $(DEPS_ALPHABETA) dummy_chess_uci
	$(CXX) $(PROFFLAGS) -fprofile-use=uci.profdata $(CXXFLAGS) -Wno-backend-plugin alphabeta.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci: $(DEPS_UCI)
	rm -rvf uci.d.profdata
	rm -vf uci.profdata
	$(CXX) $(OPTFLAGS) -fprofile-generate=uci.d.profdata -DMUTE_ERRORS $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@
	./scripts/pgo_bench.py "./dummy_chess_uci"
	$(LLVM_PROFDATA) merge -output=uci.profdata uci.d.profdata
	$(CXX) $(OPTFLAGS) -fprofile-use=uci.profdata $(CXXFLAGS) -DMUTE_ERRORS uci.cpp $(SOURCES) $(LDFLAGS) -o $@

PROFILE_USE := -fprofile-use=uci.profdata

endif

dummy_chess_bench: $(DEPS_BENCH) dummy_chess_uci
	$(CXX) $(OPTFLAGS) $(PROFILE_USE) $(CXXFLAGS) bench.cpp $(SOURCES) $(LDFLAGS) -o $@

dummy_chess_uci_dbg: $(DEPS_UCI)
	$(CXX) $(DBGFLAGS) $(CXXFLAGS) uci.cpp $(SOURCES) $(LDFLAGS) -o $@

$(LIBDUMMYCHESS): $(DEPS_SHARED)
	$(CXX) $(LIBBUILDFLAGS) $(CXXLIBFLAGS) shared_object.cpp $(SOURCES) $(LDFLAGS) -fPIC -shared -o $@

libdummychess.a: $(DEPS_STATIC)
	$(CXX) $(LIBBUILDFLAGS) $(CXXLIBFLAGS) -c shared_object.cpp $(LDFLAGS) -o shared_object.o
	$(CXX) $(LIBBUILDFLAGS) $(CXXLIBFLAGS) -c m42.cpp $(LDFLAGS) -o m42.o
	ar rcs "$@" shared_object.o m42.o

test:
	./scripts/perft_test.py

clean:
	rm -vf *.o *.gcda uci.profdata
	rm -rvf uci.d.profdata
	rm -vf dummy_chess dummy_chess_opt dummy_chess_curses dummy_chess_bench dummy_chess_alphabeta dummy_chess_uci dummy_chess_uci_dbg $(LIBDUMMYCHESS) libdummychess.a
