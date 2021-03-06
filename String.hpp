#pragma once


#include <string>
#include <vector>
#include <algorithm>
#include <iostream>


using namespace std::string_literals;


namespace str {

template <typename IterableS>
std::string join(const IterableS &iterable, std::string joinstr=", "s) {
  std::string s;
  size_t i = 0;
  for(const auto &it : iterable) {
    s += it;
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

int _add_to_vector(std::vector<std::string> &v, const char *c) {
  return _add_to_vector(v, std::string(c));
}

int _add_to_vector(std::vector<std::string> &v, const std::vector<std::string> &w) {
  return _add_to_vector(v, str::join(w, " "s));
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

} // namespace str
