#pragma once


#define NEVER_INLINE __attribute__ ((noinline))
#ifndef NDEBUG
  #define INLINE inline
  #if __APPLE__
  #define ALWAYS_INLINE __attribute__((always_inline))
  #else
  #define ALWAYS_INLINE __always_inline
  #endif
#else
  #define INLINE inline
  #define ALWAYS_INLINE INLINE
#endif

#ifndef NDEBUG
#define ALWAYS_UNROLL
#else
#define ALWAYS_UNROLL __attribute__((optimize("unroll-loops")))
#endif
