#include <vector>

#include <Bitboard.hpp>
#include <Constants.hpp>
#include <Optimizations.hpp>


struct MoveLine {
  std::vector<move_t> line;
  size_t start = 0;

  INLINE MoveLine()
  {}

  size_t size() const {
    return line.size() - start;
  }

  INLINE bool empty() const {
    return size() == 0;
  }

  INLINE void pop_back() {
    line.pop_back();
  }

  INLINE void put(move_t m) {
    line.emplace_back(m);
  }

  INLINE move_t operator[](size_t i) const {
    return line[start+i];
  }

  INLINE move_t &operator[](size_t i) {
    return line[start+i];
  }

  INLINE move_t front() const {
    return line.front();
  }

  INLINE decltype(auto) begin() const {
    return line.begin() + start;
  }

  INLINE decltype(auto) end() const {
    return line.end();
  }

  INLINE void resize(size_t new_size) {
    line.resize(start + new_size);
  }

  INLINE void shift_start() {
    assert(!empty());
    ++start;
  }

  INLINE void premove(move_t m) {
    put(m);
    shift_start();
  }

  INLINE void recall() {
    if(start > 0) {
      --start;
    }
  }

  void replace_line(const MoveLine &other) {
    for(size_t i = 0; i < other.size(); ++i) {
      if(i == size()) {
        put(other[i]);
      } else {
        operator[](i) = other[i];
      }
    }
    if(other.size() < size()) {
      resize(other.size());
    }
  }

  MoveLine branch_from_past() const {
    MoveLine mline = *this;
    while(!mline.empty()) {
      mline.pop_back();
    }
    return mline;
  }

  decltype(auto) full() const {
    return line;
  }

  void clear() {
    line.clear();
    start = 0;
  }
};
