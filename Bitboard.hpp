#pragma once


#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <climits>

#include <array>

#include <Optimizations.hpp>


// type alias: position index on bitboard
typedef uint8_t pos_t;
typedef uint16_t pos_pair_t;


// https://graphics.stanford.edu/~seander/bithacks.html
// bit-hacks from stanford graphics, to be placed here
namespace bitmask {
  const uint64_t vline = UINT64_C(0x0101010101010101);
  const uint64_t hline = UINT64_C(0xFF);

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
    return !(v & (v - 1));
  }

  template <typename T>
  inline constexpr pos_t log2_of_exp2(T v) {
    pos_t r = 0x00;
    if (v >= (UINT64_C(1) << 32)) { r += 32; v >>= 32; }
    if (v >= (UINT64_C(1) << 16)) { r += 16; v >>= 16; }
    if (v >= (UINT64_C(1) << 8 )) { r += 8 ; v >>= 8 ; }
    if (v >= (UINT64_C(1) << 4 )) { r += 4 ; v >>= 4 ; }
    if (v >= (UINT64_C(1) << 2 )) { r += 2 ; v >>= 2 ; }
    if (v >= (UINT64_C(1) << 1 )) { r += 1 ; v >>= 1 ; }
    return r;
  }

  template <typename T>
  inline constexpr pos_t count_bits(T v) {
    if(!v)return 0;
    if(bitmask::is_exp2(v))return 1;
    v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
    v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
    v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
    return (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * CHAR_BIT; // count
  }

  template <typename T>
  inline constexpr pos_t log2(T t) {
    pos_t shift = 0;
    pos_t r = 0;
    r =     (t > 0xFFFFFFFFLLU)?1<<5:0; t >>= r;
    shift = (t > 0xFFFF)       ?1<<4:0; t >>= shift; r |= shift;
    shift = (t > 0xFF)         ?1<<3:0; t >>= shift; r |= shift;
    shift = (t > 0xF)          ?1<<2:0; t >>= shift; r |= shift;
    shift = (t > 0x3)          ?1<<1:0; t >>= shift; r |= shift;
                                                     r |= (t >> 1);
    return r;
  }

  // https://www.chessprogramming.org/BitScan#DeBruijnMultiplation
  inline constexpr uint64_t lowest_bit(uint64_t v) {
    if(!v)return 0ULL;
    constexpr int index64[64] = {
        0, 47,  1, 56, 48, 27,  2, 60,
       57, 49, 41, 37, 28, 16,  3, 61,
       54, 58, 35, 52, 50, 42, 21, 44,
       38, 32, 29, 23, 17, 11,  4, 62,
       46, 55, 26, 59, 40, 36, 15, 53,
       34, 51, 20, 43, 31, 22, 10, 45,
       25, 39, 14, 33, 19, 30,  9, 24,
       13, 18,  8, 12,  7,  6,  5, 63
    };
    const uint64_t debruijn64 = UINT64_C(0x03f79d71b4cb0a89);
    assert(v);
    return 1ULL << index64[((v ^ (v-1)) * debruijn64) >> 58];
  }

  template <typename T>
  inline constexpr T highest_bit(T v) {
    if(!v)return 0ULL;
    v |= (v >>  1);
    v |= (v >>  2);
    v |= (v >>  4);
    v |= (v >>  8);
    v |= (v >> 16);
    v |= (v >> 32);
    return v - (v >> 1);
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
    if(is_exp2(mask)) {
      func(bitmask::log2_of_exp2(mask));
      return;
    }
    uint64_t x = mask;
    while(x) {
      pos_t r = bitmask::log2(x);
      func(r);
      // unset msb
      assert(x & 1ULL << r);
      x &= ~(1LLU << r);
    }
  }

  // print locations of each set bit
  NEVER_INLINE void print(uint64_t mask) {
    foreach(mask, [](pos_t p) mutable { printf("(%c, %c)\n", 'A' + (p%8), '1' + (p/8)); });
  }

  NEVER_INLINE void print_mask(uint64_t mask, pos_t markspot=0xff) {
    printf("mask: %lx\n", mask);
    char s[256];
    pos_t j = 0;
    for(pos_t i = 0; i < CHAR_BIT*sizeof(mask); ++i) {
      if(i == markspot) {
        s[j++] = 'x';
      } else {
        s[j++] = (mask & (1LLU << i)) ? '*' : '.';
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
