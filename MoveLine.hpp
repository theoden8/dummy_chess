#include <vector>

#include <Bitboard.hpp>
#include <Constants.hpp>
#include <Optimizations.hpp>


struct MoveLine {
  std::vector<move_t> line;
  size_t start = 0;
  const MoveLine *mainline = nullptr;

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

  INLINE void total_recall() {
    while(start)recall();
  }

  INLINE MoveLine full() const {
    MoveLine f = *this;
    f.start = 0;
    return f;
  }

  void replace_line(const MoveLine &other) {
    resize(other.size());
    for(size_t i = 0; i < other.size(); ++i) {
      (*this)[i] = other[i];
    }
    assert(full().size() == start + other.size());
  }

  INLINE bool is_mainline() const {
    return mainline == nullptr;
  }

  INLINE const MoveLine *get_mainline() const {
    if(is_mainline()) {
      return this;
    }
    return mainline->get_mainline();
  }

  INLINE bool find_in_mainline(move_t m) const {
    if(get_mainline() == nullptr) {
      return false;
    }
    return std::find(get_mainline()->begin(), get_mainline()->end(), m) != get_mainline()->end();
  }

  INLINE move_t front_in_mainline() const {
    if(get_mainline() == nullptr) {
      return board::nomove;
    }
    const auto &mline = get_mainline()->full();
    if(start >= mline.size()) {
      return board::nomove;
    }
    return mline[start];
  }

  MoveLine branch_from_past() const {
    MoveLine mline = *this;
    while(!mline.empty()) {
      mline.pop_back();
    }
    mline.mainline = get_mainline();
    return mline;
  }

  MoveLine get_future() const {
    MoveLine mline;
    for(auto m : *this) {
      mline.put(m);
    }
    return mline;
  }

  void clear() {
    line.clear();
    start = 0;
  }
};