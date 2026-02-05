#pragma once


#define NEVER_INLINE __attribute__ ((noinline))
#ifndef NDEBUG
  #ifndef INLINE
    #define INLINE NEVER_INLINE
  #endif
  #define ALWAYS_INLINE __attribute__((always_inline))
#else
  #ifndef INLINE
    #define INLINE inline
  #endif
  #define ALWAYS_INLINE INLINE
#endif

#ifndef NDEBUG
  #define ALWAYS_UNROLL
#else
  #define ALWAYS_UNROLL __attribute__((optimize("unroll-loops")))
#endif

#ifndef NDEBUG
  #define FLATTEN
#else
  #define FLATTEN __attribute__((flatten))
#endif

#ifdef FLAG_EXPORT
  #define NOEXPORT __attribute__((visibility("hidden")))
  #define EXPORT __attribute__((used,retain,visibility("default")))
  #define EXPORT_CLASS __attribute__((visibility("default")))
#else
  #define NOEXPORT
  #define EXPORT
  #define EXPORT_CLASS
#endif
