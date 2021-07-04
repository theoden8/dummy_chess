#pragma once


#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <climits>

#include <bit>
#include <array>
#include <vector>

#include <Optimizations.hpp>


// type alias: position index on bitboard
typedef uint8_t pos_t;
typedef uint16_t pos_pair_t;


// https://graphics.stanford.edu/~seander/bithacks.html
// bit-hacks from stanford graphics, to be placed here

// https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
// https://en.cppreference.com/w/cpp/header/bit
namespace bitmask {
  constexpr uint64_t full = ~UINT64_C(0);
  constexpr uint64_t vline = UINT64_C(0x0101010101010101);
  constexpr uint64_t hline = UINT64_C(0xFF);
  constexpr uint64_t wcheckers = UINT64_C(0x5555555555555555);
  constexpr uint64_t bcheckers = ~wcheckers;

  inline constexpr pos_pair_t _pos_pair(pos_t a, pos_t b) {
    return pos_pair_t(a) << 8 | b;
  }

  inline constexpr pos_t first(pos_pair_t p) {
    return p >> 8;
  }

  inline constexpr pos_t second(pos_pair_t p) {
    return (pos_t)p;
  }

  template <typename T>
  inline constexpr bool is_exp2(T v) {
    return std::has_single_bit(v);
  }

  template <typename T>
  inline constexpr pos_t log2_of_exp2(T v) {
    return std::countr_zero(v);
  }

  template <typename T>
  inline constexpr pos_t count_bits(T v) {
    return std::popcount(v);
  }

  template <typename T>
  inline constexpr pos_t log2(T t) {
    return std::bit_width(t) - 1;
  }

  // https://www.chessprogramming.org/BitScan#DeBruijnMultiplation
  inline constexpr uint64_t lowest_bit(uint64_t v) {
    return 1ULL << std::countr_zero(v);
  }

  template <typename T>
  inline constexpr T highest_bit(T v) {
    return std::bit_floor(v);
  }

  template <typename T>
  inline constexpr T ones_before_eq_bit(T v) {
    if(!v)return UINT64_MAX;
    assert(bitmask::is_exp2(v));
    return (v - 1) << 1;
  }

  template <typename T>
  inline constexpr T ones_before_bit(T v) {
    return ones_before_eq_bit(v >> 1);
  }

  template <typename T>
  inline constexpr T ones_after_eq_bit(T v) {
    if(!v)return UINT64_MAX;
    assert(bitmask::is_exp2(v));
    return ~(v - 1);
  }

  template <typename T>
  inline constexpr T ones_after_bit(T v) {
    return ones_after_eq_bit(v << 1);
  }

  // iterate set bits with a function F
  template <typename F>
  inline constexpr void foreach(uint64_t mask, F &&func) {
    if(!mask)return;
    while(mask) {
      const pos_t r = std::countr_zero(mask);
      func(r);
      // unset r-th bit
      assert(mask & (1ULL << r));
      mask &= ~(1LLU << r);
    }
  }

  // iterate set bits with a function F
  template <typename F>
  inline constexpr void foreach_early_stop(uint64_t mask, F &&func) {
    if(!mask)return;
    while(mask) {
      const pos_t r = std::countr_zero(mask);
      if(!func(r)) {
        break;
      }
      // unset r-th bit
      assert(mask & (1ULL << r));
      mask &= ~(1LLU << r);
    }
  }

  std::vector<pos_t> as_vector(uint64_t mask) {
    std::vector<pos_t> v(0, count_bits(mask));
    foreach(mask, [&](pos_t i) mutable -> void {
      v.emplace_back(i);
    });
    return v;
  }

  // print locations of each set bit
  NEVER_INLINE void print(uint64_t mask) {
    foreach(mask, [](pos_t p) mutable -> void {
      printf("(%c, %c)\n", 'A' + (p%8), '1' + (p/8));
    });
  }

  NEVER_INLINE void print_mask(uint64_t mask, pos_t markspot=0xff) {
    printf("mask: %lx\n", mask);
    char s[256];
    pos_t j = 0;
    for(pos_t i = 0; i < CHAR_BIT*sizeof(mask); ++i) {
      pos_t x = i % 8, y = i / 8;
      y = 8 - y - 1;
      const pos_t ind = y * 8 + x;
      if(i == markspot) {
        s[j++] = 'x';
      } else {
        s[j++] = (mask & (1LLU << ind)) ? '*' : '.';
      }
      s[j++] = ' ';
      if(i % CHAR_BIT == 7) {
        s[j++] = '\n';
      }
    }
    s[j] = '\0';
    puts(s);
  }
} // namespace bitmask
