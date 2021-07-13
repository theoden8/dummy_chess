#pragma once


#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include <Optimizations.hpp>

using namespace std::string_literals;

#ifndef NDEBUG
#define _printf printf
#else
#define _printf(...)
#endif


namespace str {

std::vector<std::string> split(const std::string &s, const std::string &sep=" "s) {
  std::vector<std::string> vs;
  size_t start_s = 0;
  for(size_t i = 0; i < s.size() - sep.size(); ++i) {
    if(s.substr(i, sep.size()) == sep) {
      vs.emplace_back(s.substr(start_s, i - start_s));
      start_s = i + sep.size();
      i = start_s - 1;
    }
  }
  if(start_s != s.size()) {
    vs.emplace_back(s.substr(start_s, s.size() - start_s));
  }
  return vs;
}

template <typename IterableS>
std::string join(const IterableS &iterable, const std::string &joinstr=", "s) {
  std::string s;
  size_t i = 0;
  for(const auto &it : iterable) {
    if constexpr(std::is_same_v<typename IterableS::value_type, std::string>) {
      s += it;
    } else {
      s += std::to_string(it);
    }
    if(i + 1 != iterable.size()) {
      s += joinstr;
    }
    ++i;
  }
  return s;
}

template <typename... Ts>
void _unroll_(Ts...){}

template <typename Elem>
int _add_to_vector(std::vector<std::string> &v, const Elem &s) {
  v.emplace_back(std::to_string(s));
  return 1;
}

int _add_to_vector(std::vector<std::string> &v, const std::string &s) {
  v.emplace_back(s);
  return 1;
}

int _add_to_vector(std::vector<std::string> &v, char *c) {
  return _add_to_vector(v, std::string(c));
}

int _add_to_vector(std::vector<std::string> &v, const char *c) {
  return _add_to_vector(v, std::string(c));
}

int _add_to_vector(std::vector<std::string> &v, const std::vector<std::string> &w) {
  return _add_to_vector(v, str::join(w, " "s));
}

template <typename... Elem>
void perror(const Elem & ...s) {
  std::vector<std::string> v;
  _unroll_(_add_to_vector(v, s)...);
#ifndef __clang__
  std::reverse(v.begin(), v.end());
#endif
  std::cerr << str::join(v, " "s) << std::endl;
}

template <typename... Elem>
void print(const Elem & ...s) {
  std::vector<std::string> v;
  _unroll_(_add_to_vector(v, s)...);
#ifndef __clang__
  std::reverse(v.begin(), v.end());
#endif
  std::cout << str::join(v, " "s) << std::endl;
}

template <typename... Elem>
INLINE void pdebug(const Elem & ...s) {
#ifndef NDEBUG
  std::vector<std::string> v;
  _unroll_(_add_to_vector(v, s)...);
#ifndef __clang__
  std::reverse(v.begin(), v.end());
#endif
  std::cout << str::join(v, " "s) << std::endl;
#endif
}

} // namespace str
