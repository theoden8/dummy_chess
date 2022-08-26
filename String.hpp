#pragma once


#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <concepts>
#include <string_view>

#ifndef FLAG_STDRANGES
#include <ranges>
#endif

#include <Optimizations.hpp>

using namespace std::string_literals;

#ifndef NDEBUG
#define _printf printf
#else
#define _printf(...)
#endif


namespace str {

template <typename T> concept Stringable = requires (T a) { { std::string(a) }; };
template <typename T> concept ToStringable = requires (T a) { { std::to_string(a) }; };

//#ifndef FLAG_STDRANGES
//decltype(auto) split(const std::string_view &s, const std::string_view &sep=" "s) {
//  return s | std::views::split(sep);
//}
//#else
//decltype(auto) split(const std::string_view &s, const std::string_view &sep=" "s) {
//  std::vector<std::string> vs;
//  size_t start_s = 0;
//  for(size_t i = 0; i < s.size() - sep.size(); ++i) {
//    if(s.substr(i, sep.size()) == sep) {
//      vs.emplace_back(s.substr(start_s, i - start_s));
//      start_s = i + sep.size();
//      i = start_s - 1;
//    }
//  }
//  if(start_s != s.size()) {
//    vs.emplace_back(s.substr(start_s, s.size() - start_s));
//  }
//  return vs;
//}
//#endif

template <typename ContainerT>
decltype(auto) join(const ContainerT &iterable, const std::string_view &joinstr=", "s)
  requires Stringable<typename ContainerT::value_type> || ToStringable<typename ContainerT::value_type>
{
  using T = typename ContainerT::value_type;
  std::string s;
  size_t i = 0;
  for(const auto &it : iterable) {
    if constexpr(Stringable<T>) {
      s += it;
    } else if constexpr(ToStringable<T>) {
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

template <typename T>
std::string convert_to_string(const T &s) requires Stringable<T> {
  return s;
}

template <typename T>
std::string convert_to_string(const T &s) requires ToStringable<T> {
  return std::to_string(s);
}

std::string convert_to_string(const char *s) {
  return std::string(s);
}

std::string convert_to_string(char *s) {
  return std::string(s);
}

template <typename T>
std::string convert_to_string(const std::vector<T> &vs) {
  return str::join(vs, " "s);
}

template <typename... Elem>
void perror(const Elem & ...s) {
  // unpacking order in initializer lists is defined
  std::vector<std::string> v = { convert_to_string(s)... };
  std::cerr << str::join(v, " "s) << std::endl;
}

template <typename... Elem>
void print(const Elem & ...s) {
  // unpacking order in initializer lists is defined
  std::vector<std::string> v = { convert_to_string(s)... };
  std::cout << str::join(v, " "s) << std::endl;
}

template <typename... Elem>
INLINE void pdebug(const Elem & ...s) {
#ifndef NDEBUG
  std::vector<std::string> v = { convert_to_string(s)... };
  std::cout << str::join(v, " "s) << std::endl;
#endif
}

} // namespace str
