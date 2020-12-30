OPTFLAGS = -g3
CXXFLAGS = $(OPTFLAGS) -std=c++17 -I. -fopt-info -Wall -Wextra $(shell pkgconf --libs ncurses ncursesw)
LDFLAGS = $(shell pkgconf --libs ncurses ncursesw)
HPPFILES = $(wildcard *.hpp)

all : dummy_chess

dummy_chess: main.cpp $(HPPFILES) Makefile
	$(CXX) $(CXXFLAGS) main.cpp $(LDFLAGS) -o dummy_chess

clean:
	rm -vf dummy_chess
