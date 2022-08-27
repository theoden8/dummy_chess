#pragma once

#ifdef FLAG_JEMALLOC_EXTERNAL
#include <jemalloc/jemalloc.h>
#endif

#include <vector>

#include <Optimizations.hpp>
#include <Bitmask.hpp>
#include <Piece.hpp>


class Board;


struct MoveLine {
  std::vector<move_t> line;
  size_t start = 0;
  bool mainline = true;
  bool tb = false;

  INLINE MoveLine()
  {}

  explicit INLINE MoveLine(const std::vector<move_t> &line, bool mainline=true, bool shrink=false, bool tb=false):
    line(line), mainline(mainline), tb(tb)
  {
    if(!shrink) {
      this->line.reserve(16);
    } else {
      this->line.shrink_to_fit();
    }
  }

  INLINE size_t size() const {
    return line.size() - start;
  }

  INLINE bool empty() const {
    return self.size() == 0;
  }

  INLINE bool operator==(const MoveLine &other) const {
    return self.size() == other.size() && self.startswith(other);
  }

  INLINE bool startswith(const MoveLine &other) const {
    for(size_t i = 0; i < std::min(size(), other.size()); ++i) {
      if(self[i] != other[i])return false;
    }
    return true;
  }

  INLINE void pop_back() {
    line.pop_back();
  }

  INLINE void put(move_t m) {
    tb = false;
    line.emplace_back(m);
  }

  INLINE move_t operator[](size_t i) const {
    return line[start+i];
  }

  INLINE move_t &operator[](size_t i) {
    return line[start+i];
  }

  INLINE move_t front() const {
    return self.empty() ? board::nullmove : line[start];
  }

  INLINE move_t back() const {
    return self.empty() ? board::nullmove : line.back();
  }

  INLINE decltype(auto) begin() const {
    return line.begin() + start;
  }

  INLINE decltype(auto) end() const {
    return line.end();
  }

  INLINE void resize(size_t new_size) {
    tb = false;
    line.resize(start + new_size);
  }

  INLINE void shift_start() {
    assert(!self.empty());
    ++start;
  }

  INLINE void premove(move_t m) {
    tb = false;
    if(self.empty()) {
      self.put(m);
    } else if(m == self.front()) {
      ;
    } else {
      self.resize(1);
      self[0] = m;
    }
    self.shift_start();
  }

  INLINE void recall() {
    if(start > 0)--start;
  }

  INLINE void recall_n(size_t n) {
    assert(n < start);
    start -= n;
  }

  INLINE void total_recall() {
    while(start)self.recall();
  }

  INLINE MoveLine full() const {
    MoveLine f = self;
    f.start = 0;
    return f;
  }

  INLINE void set_mainline(bool ismainline) {
    mainline = ismainline;
  }

  INLINE void replace_line(const MoveLine &other) {
    self.resize(other.size());
    tb = other.tb;
    for(size_t i = 0; i < other.size(); ++i) {
      self[i] = other[i];
    }
    assert(line.size() == start + other.size());
  }

  template <size_t N>
  INLINE void draft_line(const std::array<move_t, N> &m_hint) {
    tb = false;
    size_t sz = 0;
    for(sz = 0; sz < m_hint.size(); ++sz) {
      if(m_hint[sz] == board::nullmove)break;
      self.premove(m_hint[sz]);
    }
    assert(self.start >= sz);
    self.start -= sz;
  }

  INLINE bool is_mainline() const {
    return mainline;
  }

  INLINE bool find(move_t m) const {
    return std::find(self.begin(), self.end(), m) != self.end();
  }

  INLINE bool find_even(move_t m, size_t start_index) const {
    if(line.empty()) {
      return false;
    }
    for(size_t i = start_index; i < line.size(); i += 2) {
      if(line[i] == m)return true;
    }
    return false;
  }

  INLINE MoveLine get_future() const {
    return MoveLine(std::vector<move_t>(self.begin(), self.end()), mainline, true, tb);
  }

  INLINE MoveLine get_past() const {
    return MoveLine(std::vector<move_t>(line.begin(), self.begin()), mainline, true, false);
  }

  INLINE move_t get_previous_move() const {
    if(start == 0)return board::nullmove;
    if(line[start - 1] != board::nullmove) {
      return line[start - 1];
    }
    if(start < 3)return board::nullmove;
    return line[start - 3];
  }

  INLINE move_t get_next_move() const {
    if(self.size() < 2)return board::nullmove;
    return line[start + 1];
  }

  INLINE MoveLine as_past() const {
    MoveLine mline = self;
    mline.start = line.size();
    return mline;
  }

  INLINE MoveLine branch_from_past(move_t m=board::nullmove) const {
    if(self.front() == m && m != board::nullmove) {
      MoveLine mline = self;
      mline.tb = false;
      return mline;
    }
    MoveLine mline = self.get_past();
    mline.start = start;
    mline.mainline = false;
    return mline;
  }

  INLINE void clear() {
    line.resize(start);
  }

  NEVER_INLINE std::string pgn(Board &engine) const;
  NEVER_INLINE std::string pgn_full(Board &engine) const;
};
