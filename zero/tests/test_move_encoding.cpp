// Tests for MoveEncoding.hpp: encode/decode, roundtrip, specific moves.

#include <gtest/gtest.h>

#include <MoveEncoding.hpp>

TEST(MoveEncoding, PolicySize) {
    EXPECT_EQ(dc0::POLICY_SIZE, 4672);
}

TEST(MoveEncoding, ValidMoveCount) {
    int count = 0;
    for (int i = 0; i < dc0::POLICY_SIZE; ++i) {
        if (dc0::is_valid_policy_index(i)) ++count;
    }
    EXPECT_EQ(count, 1924);
}

TEST(MoveEncoding, EncodeE2E4) {
    // e2=E2=12, e4=E4=28
    int idx = dc0::encode_move(12, 28, EMPTY);
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, dc0::POLICY_SIZE);
    EXPECT_EQ(idx, 877);

    dc0::DecodedMove dm = dc0::decode_move(idx);
    EXPECT_EQ(dm.from_sq, 12);
    EXPECT_EQ(dm.to_sq, 28);
    EXPECT_EQ(dm.promo, EMPTY);
}

TEST(MoveEncoding, EncodeKnight) {
    // Ng1f3: G1=6, F3=21
    int idx = dc0::encode_move(6, 21, EMPTY);
    EXPECT_EQ(idx, 501);

    dc0::DecodedMove dm = dc0::decode_move(idx);
    EXPECT_EQ(dm.from_sq, 6);
    EXPECT_EQ(dm.to_sq, 21);
}

TEST(MoveEncoding, QueenPromotion) {
    // e7e8=Q: E7=52, E8=60
    int idx = dc0::encode_move(52, 60, QUEEN);
    EXPECT_EQ(idx, 3796);

    dc0::DecodedMove dm = dc0::decode_move(idx);
    EXPECT_EQ(dm.from_sq, 52);
    EXPECT_EQ(dm.to_sq, 60);
    EXPECT_EQ(dm.promo, QUEEN);
}

TEST(MoveEncoding, KnightUnderpromotion) {
    // e7e8=N: E7=52, E8=60
    int idx = dc0::encode_move(52, 60, KNIGHT);
    EXPECT_EQ(idx, 3863);

    dc0::DecodedMove dm = dc0::decode_move(idx);
    EXPECT_EQ(dm.from_sq, 52);
    EXPECT_EQ(dm.to_sq, 60);
    EXPECT_EQ(dm.promo, KNIGHT);
}

TEST(MoveEncoding, MoveTRoundtrip) {
    // e2e4: move_t = (12 << 8) | 28
    move_t m = bitmask::_pos_pair(12, 28);
    int idx = dc0::encode_move_t(m);
    move_t back = dc0::decode_to_move_t(idx);
    EXPECT_EQ(m, back);

    // e7e8=Q
    m = bitmask::_pos_pair(52, 60 | board::PROMOTE_QUEEN);
    idx = dc0::encode_move_t(m);
    back = dc0::decode_to_move_t(idx);
    EXPECT_EQ(m, back);

    // e7e8=N
    m = bitmask::_pos_pair(52, 60 | board::PROMOTE_KNIGHT);
    idx = dc0::encode_move_t(m);
    back = dc0::decode_to_move_t(idx);
    EXPECT_EQ(m, back);
}

TEST(MoveEncoding, AllValidRoundtrip) {
    int roundtrip_ok = 0;
    for (int i = 0; i < dc0::POLICY_SIZE; ++i) {
        if (!dc0::is_valid_policy_index(i)) continue;
        dc0::DecodedMove dm = dc0::decode_move(i);
        int re_encoded = dc0::encode_move(dm.from_sq, dm.to_sq, dm.promo);
        EXPECT_EQ(re_encoded, i) << "roundtrip failed for idx=" << i;
        if (re_encoded == i) ++roundtrip_ok;
    }
    EXPECT_EQ(roundtrip_ok, 1924);
}
