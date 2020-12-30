OPTFLAGS = -g3
CXXFLAGS = $(OPTFLAGS) -std=c++17 -I. -fopt-info -Wall -Wextra
HPPFILES = $(wildcard *.hpp)


all : dummy_chess

dummy_chess: main.cpp $(HPPFILES) Makefile
	$(CXX) $(CXXFLAGS) main.cpp -o dummy_chess

clean:
	rm -vf dummy_chess
