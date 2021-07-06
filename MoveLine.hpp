#pragma once


#include <vector>

#include <Optimizations.hpp>
#include <Bitmask.hpp>
#include <Bitboard.hpp>


struct MoveLine {
  std::vector<move_t> line;
  size_t start = 0;
  const MoveLine *mainline = nullptr;

  INLINE MoveLine()
  {}

  explicit INLINE MoveLine(const std::vector<move_t> &line, const MoveLine *mainline=nullptr):
    line(line), mainline(mainline)
  {
    this->line.reserve(16);
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

  void set_mainline(const MoveLine *other) {
    mainline = other;
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

  INLINE bool find(move_t m, size_t start_index) const {
    if(line.empty()) {
      return false;
    }
    for(size_t i = start_index; i < line.size(); i += 2) {
      if(line[i] == m)return true;
    }
    return false;
  }

  INLINE bool find_in_mainline(move_t m) const {
    return get_mainline() != nullptr && get_mainline()->find(m, start & 1);
  }

  INLINE move_t front_in_mainline() const {
    if(get_mainline() == nullptr) {
      return board::nomove;
    }
    const auto &m_line = get_mainline()->line;
    if(start >= m_line.size()) {
      return board::nomove;
    }
    return m_line[start];
  }

  INLINE MoveLine get_future() const {
    return MoveLine(std::vector<move_t>(begin(), end()), get_mainline());
  }

  INLINE MoveLine get_past() const {
    return MoveLine(std::vector<move_t>(line.begin(), begin()), get_mainline());
  }

  INLINE MoveLine as_past() const {
    MoveLine mline = *this;
    mline.start = line.size();
    return mline;
  }

  INLINE MoveLine branch_from_past() const {
    MoveLine mline = get_past();
    mline.start = start;
    if(get_mainline() != nullptr) {
      mline.mainline = get_mainline();
    } else {
      mline.mainline = this;
    }
    return mline;
  }

  INLINE void clear() {
    line.resize(start);
  }
};
