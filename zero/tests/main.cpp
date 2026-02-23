// dc0 test main: initializes engine globals (M42, Zobrist) before all tests.
//
// We forward-declare the init functions rather than including Zobrist.hpp and
// m42.h here, because those headers contain non-inline function definitions
// that would cause multiple-definition link errors when combined with other
// TUs that also include them (via Board.hpp etc.).

#include <gtest/gtest.h>

#include <cstddef>

// Forward declarations — defined in m42.cpp and Zobrist.hpp (included by other TUs).
namespace M42 { void init(); }
namespace zobrist { void init(size_t); }

class DC0Environment : public ::testing::Environment {
public:
    void SetUp() override {
        M42::init();
        zobrist::init(1 << 16);
    }
};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new DC0Environment);
    return RUN_ALL_TESTS();
}
