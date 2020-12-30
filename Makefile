OPTFLAGS = -g3
CXXFLAGS = $(OPTFLAGS) -std=c++17


all : dummy_chess


dummy_chess: main.cpp Makefile
	$(CXX) $(CXXFLAGS) main.cpp -o dummy_chess


clean:
	rm -vf dummy_chess
