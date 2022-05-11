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

  INLINE MoveLine()
  {}

  explicit INLINE MoveLine(const std::vector<move_t> &line, bool mainline=true, bool shrink=false):
    line(line), mainline(mainline)
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
    return size() == 0;
  }

  INLINE bool operator==(const MoveLine &tline) const {
    return size() == tline.size() && startswith(tline);
  }

  INLINE bool startswith(const MoveLine &tline) const {
    for(size_t i = 0; i < std::min(size(), tline.size()); ++i) {
      if((*this)[i] != tline[i])return false;
    }
    return true;
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
    return empty() ? board::nullmove : line[start];
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
    if(empty()) {
      put(m);
    } else if(m == front()) {
      ;
    } else {
      clear();
      put(m);
    }
    shift_start();
  }

  INLINE void recall() {
    if(start > 0)--start;
  }

  INLINE void total_recall() {
    while(start)recall();
  }

  INLINE MoveLine full() const {
    MoveLine f = *this;
    f.start = 0;
    return f;
  }

  INLINE void set_mainline(bool ismainline) {
    mainline = ismainline;
  }

  INLINE void replace_line(const MoveLine &other) {
    resize(other.size());
    for(size_t i = 0; i < other.size(); ++i) {
      (*this)[i] = other[i];
    }
    assert(line.size() == start + other.size());
  }

  INLINE void draft(move_t m) {
    premove(m);
    recall();
  }

  template <size_t N>
  INLINE void draft_line(const std::array<move_t, N> &m_hint) {
    size_t sz = 0;
    for(sz = 0; sz < m_hint.size(); ++sz) {
      if(m_hint[sz] == board::nullmove)break;
      premove(m_hint[sz]);
    }
    for(size_t i = 0; i < sz; ++i) {
      recall();
    }
  }

  INLINE bool is_mainline() const {
    return mainline;
  }

  INLINE bool find(move_t m) const {
    return std::find(begin(), end(), m) != end();
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
    return MoveLine(std::vector<move_t>(begin(), end()), mainline, true);
  }

  INLINE MoveLine get_past() const {
    return MoveLine(std::vector<move_t>(line.begin(), begin()), mainline);
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
    if(size() < 2)return board::nullmove;
    return line[start + 1];
  }

  INLINE MoveLine as_past() const {
    MoveLine mline = *this;
    mline.start = line.size();
    return mline;
  }

  INLINE MoveLine branch_from_past(move_t m=board::nullmove) const {
    if(front() == m && m != board::nullmove) {
      return *this;
    }
    MoveLine mline = get_past();
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
