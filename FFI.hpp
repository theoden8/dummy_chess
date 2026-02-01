#pragma once

#include <cstdio>

#include <string>

#include "Optimizations.hpp"

struct FFIString {
  EXPORT static const char *to_cstring(const std::string &s) {
    static char buf[255];
    snprintf(buf, 255, "%s", s.c_str());
    return buf;
  }

  EXPORT static void show_cstring(const std::string &s) {
    printf("%s\n", s.c_str());
    fflush(stdout);
  }

  EXPORT static std::string make_string(const char *s) {
    return std::string(s);
  }
};
