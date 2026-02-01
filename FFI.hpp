#pragma once

#include <cstdio>

#include <string>

struct FFIString {
  __attribute__((used)) static const char *to_cstring(const std::string &s) {
    static char buf[255];
    snprintf(buf, 255, "%s", s.c_str());
    return buf;
  }

  __attribute__((used)) static void show_cstring(const std::string &s) {
    printf("%s\n", s.c_str());
    fflush(stdout);
  }
};
