"""
Tests for HalfKAv2 feature extraction.

Tests the C++ HalfKAv2 feature extraction against a reference
implementation using python-chess.

HalfKAv2 uses:
- 12 king buckets (based on king position, with files e-h mirrored to a-d)
- Horizontal mirroring when king is on kingside
- Feature size: 12 * 641 + 1 = 7693
"""

import chess
import numpy as np
import pytest

import dummy_chess


# King bucket table: maps king square (0-63) to bucket (0-11)
# For files e-h, we mirror to a-d first
KING_BUCKET = [
    # Rank 1 (a1-d1): buckets 0-3
    0,
    1,
    2,
    3,
    # Rank 2 (a2-d2): buckets 4-5
    4,
    4,
    5,
    5,
    # Rank 3 (a3-d3): buckets 6-7
    6,
    6,
    7,
    7,
    # Rank 4 (a4-d4): bucket 8
    8,
    8,
    8,
    8,
    # Rank 5 (a5-d5): bucket 9
    9,
    9,
    9,
    9,
    # Rank 6 (a6-d6): bucket 10
    10,
    10,
    10,
    10,
    # Ranks 7-8 (a7-d8): bucket 11
    11,
    11,
    11,
    11,
]

# Piece type to index mapping: [white_index, black_index]
PIECE_TO_INDEX = {
    0: [0, 1],  # Pawn
    1: [2, 3],  # Knight
    2: [4, 5],  # Bishop
    3: [6, 7],  # Rook
    4: [8, 9],  # Queen
}


def get_bucket_and_mirror(king_sq: int) -> tuple[int, bool]:
    """Get king bucket and whether to mirror horizontally."""
    file = king_sq % 8
    rank = king_sq // 8
    mirror = file >= 4
    if mirror:
        file = 7 - file  # Mirror e-h to d-a
    bucket_idx = rank * 4 + file
    if bucket_idx >= 32:
        bucket_idx = 31
    return KING_BUCKET[bucket_idx], mirror


def get_halfkav2_features_reference(fen: str) -> tuple[list[int], list[int], int]:
    """Reference HalfKAv2 feature extraction using python-chess."""
    board = chess.Board(fen)
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    # Get king buckets and mirror flags
    w_bucket, w_mirror = get_bucket_and_mirror(wk)
    b_bucket, b_mirror = get_bucket_and_mirror(63 - bk)  # Flip for black's perspective

    white_feats, black_feats = [], []
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None or pc.piece_type == chess.KING:
            continue
        pt = pc.piece_type - 1  # Convert to 0-4 range
        is_white = pc.color

        w_idx = 0 if is_white else 1
        b_idx = 1 if is_white else 0

        # Apply horizontal mirroring if king is on kingside
        w_piece_sq = sq ^ 7 if w_mirror else sq  # XOR with 7 mirrors file
        b_piece_sq_flipped = 63 - sq  # Vertical flip for black
        b_piece_sq = b_piece_sq_flipped ^ 7 if b_mirror else b_piece_sq_flipped

        # Feature index: bucket * 641 + piece_index * 64 + piece_sq + 1
        white_feat = w_bucket * 641 + PIECE_TO_INDEX[pt][w_idx] * 64 + w_piece_sq + 1
        black_feat = b_bucket * 641 + PIECE_TO_INDEX[pt][b_idx] * 64 + b_piece_sq + 1

        white_feats.append(white_feat)
        black_feats.append(black_feat)

    return white_feats, black_feats, 0 if board.turn else 1


# Test FENs covering various positions
TEST_FENS = [
    # Starting position
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "starting"),
    # After 1. e4
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e4"),
    # Italian Game
    (
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "italian",
    ),
    # Complex middlegame (Kiwipete)
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "kiwipete",
    ),
    # Simple endgame
    ("8/8/8/8/8/5k2/8/4K2R w - - 0 1", "KR_vs_K"),
    # Endgame with pawns
    ("8/5k2/8/8/8/8/5P2/4K3 w - - 0 1", "KP_vs_K"),
    # Black to move
    ("8/8/8/4k3/8/8/4P3/4K3 b - - 0 1", "black_to_move"),
    # Many pieces
    (
        "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1BP2/PP4PP/R2QKB1R w KQ - 0 9",
        "sicilian_dragon",
    ),
    # Minimal position (just kings)
    ("8/8/8/4k3/8/8/8/4K3 w - - 0 1", "bare_kings"),
    # Asymmetric material
    ("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1", "white_full_black_king"),
    # King on kingside (e-h files) - tests mirroring
    (
        "rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 1",
        "both_kings_kingside",
    ),
    # King on queenside
    (
        "r3kbnr/ppp1pppp/2nq4/3p4/3P4/2NQ4/PPP1PPPP/R3KBNR w KQkq - 0 1",
        "kings_queenside",
    ),
    # White king on g1, black king on c8 (asymmetric mirroring)
    ("2k5/8/8/8/8/8/8/6K1 w - - 0 1", "asymmetric_king_positions"),
]


class TestHalfKAv2Features:
    """Tests for HalfKAv2 feature extraction."""

    @pytest.mark.parametrize("fen,name", TEST_FENS)
    def test_features_match_reference(self, fen: str, name: str):
        """Test that C++ implementation matches reference for various positions."""
        # Get reference features
        w_ref, b_ref, stm_ref = get_halfkav2_features_reference(fen)

        # Get C++ features via batch API
        compressed = dummy_chess.compress_fen(fen)
        w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkav2_features_batch(
            [compressed], flip=False
        )

        w_cpp = list(w_idx)
        b_cpp = list(b_idx)
        stm_cpp = stm[0]

        # Sort features since order may differ
        assert sorted(w_cpp) == sorted(w_ref), f"White features mismatch for {name}"
        assert sorted(b_cpp) == sorted(b_ref), f"Black features mismatch for {name}"
        assert stm_cpp == stm_ref, f"STM mismatch for {name}"

    def test_feature_count(self):
        """Test that feature count matches piece count."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        compressed = dummy_chess.compress_fen(fen)
        w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkav2_features_batch(
            [compressed]
        )

        # Starting position has 30 non-king pieces
        assert len(w_idx) == 30
        assert len(b_idx) == 30

    def test_feature_bounds(self):
        """Test that all features are within valid HalfKAv2 range."""
        max_feature = dummy_chess.HALFKAV2_SIZE  # 7693

        for fen, name in TEST_FENS:
            compressed = dummy_chess.compress_fen(fen)
            w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkav2_features_batch(
                [compressed]
            )

            for feat in list(w_idx) + list(b_idx):
                assert 1 <= feat < max_feature, (
                    f"Feature {feat} out of bounds for {name}"
                )

    def test_empty_position(self):
        """Test handling of position with only kings."""
        fen = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
        compressed = dummy_chess.compress_fen(fen)
        w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkav2_features_batch(
            [compressed]
        )

        assert len(w_idx) == 0
        assert len(b_idx) == 0
        assert stm[0] == 0  # White to move

    def test_black_to_move(self):
        """Test STM is correct for black to move."""
        fen = "8/8/8/4k3/8/8/4P3/4K3 b - - 0 1"
        compressed = dummy_chess.compress_fen(fen)
        w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkav2_features_batch(
            [compressed]
        )

        assert stm[0] == 1  # Black to move

    def test_flip_swaps_perspectives(self):
        """Test that flip=True swaps white and black features."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        compressed = dummy_chess.compress_fen(fen)

        # Without flip
        w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkav2_features_batch(
            [compressed], flip=False
        )
        w_normal = sorted(list(w_idx))
        b_normal = sorted(list(b_idx))
        stm_normal = stm[0]

        # With flip
        w_idx_f, w_off_f, b_idx_f, b_off_f, stm_f = (
            dummy_chess.get_halfkav2_features_batch([compressed], flip=True)
        )
        w_flipped = sorted(list(w_idx_f))
        b_flipped = sorted(list(b_idx_f))
        stm_flipped = stm_f[0]

        # Flipped white should equal normal black and vice versa
        assert w_flipped == b_normal, "Flipped white should equal normal black"
        assert b_flipped == w_normal, "Flipped black should equal normal white"
        assert stm_flipped == 1 - stm_normal, "STM should be flipped"

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        fens = [fen for fen, _ in TEST_FENS[:5]]
        compressed = [dummy_chess.compress_fen(f) for f in fens]

        w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkav2_features_batch(
            compressed
        )

        # Check offsets are valid
        assert len(w_off) == len(fens)
        assert len(b_off) == len(fens)
        assert len(stm) == len(fens)

        # Check each position individually
        for i, fen in enumerate(fens):
            w_ref, b_ref, stm_ref = get_halfkav2_features_reference(fen)

            # Extract features for this position from batch
            w_start = w_off[i]
            w_end = w_off[i + 1] if i + 1 < len(fens) else len(w_idx)
            b_start = b_off[i]
            b_end = b_off[i + 1] if i + 1 < len(fens) else len(b_idx)

            w_batch = sorted(list(w_idx[w_start:w_end]))
            b_batch = sorted(list(b_idx[b_start:b_end]))

            assert w_batch == sorted(w_ref), f"White features mismatch at position {i}"
            assert b_batch == sorted(b_ref), f"Black features mismatch at position {i}"
            assert stm[i] == stm_ref, f"STM mismatch at position {i}"


class TestKingBuckets:
    """Tests for king bucket assignment."""

    def test_bucket_assignment(self):
        """Test that king buckets are assigned correctly."""
        # King on a1 should be bucket 0
        bucket, mirror = get_bucket_and_mirror(0)
        assert bucket == 0
        assert not mirror

        # King on h1 mirrors to a1 -> bucket 0
        bucket, mirror = get_bucket_and_mirror(7)
        assert bucket == 0
        assert mirror

        # King on e1 mirrors to d1 -> bucket 3
        bucket, mirror = get_bucket_and_mirror(4)
        assert bucket == 3
        assert mirror

        # King on d1 should be bucket 3, no mirror
        bucket, mirror = get_bucket_and_mirror(3)
        assert bucket == 3
        assert not mirror

        # King on g1 mirrors to b1 -> bucket 1
        bucket, mirror = get_bucket_and_mirror(6)
        assert bucket == 1
        assert mirror

        # King on b1 should be bucket 1, no mirror
        bucket, mirror = get_bucket_and_mirror(1)
        assert bucket == 1
        assert not mirror

    def test_mirroring_symmetry(self):
        """Test that mirrored positions produce consistent features."""
        # Position with white king on g1 and black king on e8
        fen1 = "4k3/8/8/8/8/8/8/6K1 w - - 0 1"
        # Position with white king on b1 (mirrored) and black king on e8
        fen2 = "4k3/8/8/8/8/8/8/1K6 w - - 0 1"

        compressed1 = dummy_chess.compress_fen(fen1)
        compressed2 = dummy_chess.compress_fen(fen2)

        w1, _, b1, _, _ = dummy_chess.get_halfkav2_features_batch([compressed1])
        w2, _, b2, _, _ = dummy_chess.get_halfkav2_features_batch([compressed2])

        # Both positions have only kings, so features should be empty
        assert len(w1) == 0
        assert len(w2) == 0


class TestHalfKAv2Size:
    """Tests for HalfKAv2 constants."""

    def test_size_constant(self):
        """Test that HALFKAV2_SIZE is correct."""
        # 12 buckets * 641 (10 piece types * 64 squares + 1) + 1
        expected = 12 * 641 + 1
        assert dummy_chess.HALFKAV2_SIZE == expected
        assert dummy_chess.HALFKAV2_SIZE == 7693

    def test_halfkp_vs_halfkav2_size(self):
        """Test relative sizes of HalfKP and HalfKAv2."""
        # HalfKP: 64 king squares * 641 + 1 = 41025
        # HalfKAv2: 12 king buckets * 641 + 1 = 7693
        assert dummy_chess.HALFKP_SIZE == 41025
        assert dummy_chess.HALFKAV2_SIZE == 7693
        assert dummy_chess.HALFKP_SIZE > dummy_chess.HALFKAV2_SIZE
