#pragma once


#define NEVER_INLINE __attribute__ ((noinline))
#ifndef NDEBUG
  #define INLINE NEVER_INLINE
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

#ifdef FLAG_EXPORT
  #define NOEXPORT __attribute__((visibility("hidden")))
  #define EXPORT __attribute__((used,retain,visibility("default")))
  #define EXPORT_CLASS __attribute__((visibility("default")))
#else
  #define NOEXPORT __attribute__((visibility("hidden")))
  #define EXPORT NOEXPORT
  #define EXPORT_CLASS NOEXPORT
#endif
