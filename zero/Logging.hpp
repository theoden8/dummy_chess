#pragma once

// dc0 logging: lightweight, header-only, configurable verbosity.
//
// Usage:
//   dc0::log::set_level(dc0::log::Level::DEBUG);
//   DC0_LOG_INFO("Training generation %d, %d games", gen, n_games);
//   DC0_LOG_DEBUG("MCTS leaf value=%.4f", value);
//
// Log levels: DEBUG < INFO < WARN < ERROR < NONE
// Default level: INFO
// Output goes to stderr with timestamps.

#include <cstdio>
#include <cstdarg>
#include <chrono>

namespace dc0 {
namespace log {

enum class Level : int {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERROR = 3,
    NONE  = 4,
};

namespace detail {

inline Level& min_level() {
    static Level level = Level::INFO;
    return level;
}

inline auto& start_time() {
    static auto t = std::chrono::steady_clock::now();
    return t;
}

inline double elapsed_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time()).count();
}

inline const char* level_tag(Level level) {
    switch (level) {
        case Level::DEBUG: return "DEBUG";
        case Level::INFO:  return "INFO ";
        case Level::WARN:  return "WARN ";
        case Level::ERROR: return "ERROR";
        default:           return "?????";
    }
}

inline void log_msg(Level level, const char* file, int line, const char* fmt, ...) {
    if (level < min_level()) return;

    double t = elapsed_seconds();
    fprintf(stderr, "[%8.3f %s] ", t, level_tag(level));

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    // Append file:line for DEBUG and WARN/ERROR
    if (level == Level::DEBUG || level >= Level::WARN) {
        // Extract filename from path
        const char* base = file;
        for (const char* p = file; *p; ++p) {
            if (*p == '/') base = p + 1;
        }
        fprintf(stderr, "  [%s:%d]", base, line);
    }

    fputc('\n', stderr);
}

} // namespace detail

inline void set_level(Level level) { detail::min_level() = level; }
inline Level get_level() { return detail::min_level(); }

// Reset the clock (call at program start if you want t=0 to match main())
inline void reset_clock() { detail::start_time() = std::chrono::steady_clock::now(); }

} // namespace log
} // namespace dc0

#define DC0_LOG_DEBUG(fmt, ...) \
    dc0::log::detail::log_msg(dc0::log::Level::DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define DC0_LOG_INFO(fmt, ...) \
    dc0::log::detail::log_msg(dc0::log::Level::INFO, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define DC0_LOG_WARN(fmt, ...) \
    dc0::log::detail::log_msg(dc0::log::Level::WARN, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define DC0_LOG_ERROR(fmt, ...) \
    dc0::log::detail::log_msg(dc0::log::Level::ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
