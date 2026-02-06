"""
Tests for HalfKP feature extraction.

Tests the numba-optimized feature extraction against a reference
implementation using python-chess.
"""

import chess
import numpy as np
import pytest

from train import (
    PIECE_TO_INDEX,
    get_halfkp_features,
    get_halfkp_features_np,
    parse_fen_fast,
)


# Reference implementation using python-chess (known correct)
def get_halfkp_features_reference(fen: str) -> tuple[list[int], list[int], int]:
    """Reference HalfKP feature extraction using python-chess."""
    board = chess.Board(fen)
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)

    white_feats, black_feats = [], []
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None or pc.piece_type == chess.KING:
            continue
        pt = pc.piece_type - 1
        is_white = pc.color

        white_feats.append(
            wk * 641 + PIECE_TO_INDEX[pt][0 if is_white else 1] * 64 + sq + 1
        )
        black_feats.append(
            (63 - bk) * 641
            + PIECE_TO_INDEX[pt][1 if is_white else 0] * 64
            + (63 - sq)
            + 1
        )

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
]


class TestParsefen:
    """Tests for the fast FEN parser."""

    def test_starting_position(self):
        """Test parsing starting position."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        (
            piece_squares,
            piece_types,
            piece_colors,
            n_pieces,
            wk,
            bk,
            white_to_move,
        ) = parse_fen_fast(fen)

        assert n_pieces == 30  # 32 pieces - 2 kings
        assert wk == 4  # e1
        assert bk == 60  # e8
        assert white_to_move is True

    def test_black_to_move(self):
        """Test parsing position where black is to move."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        *_, white_to_move = parse_fen_fast(fen)
        assert white_to_move is False

    def test_endgame(self):
        """Test parsing simple endgame position."""
        fen = "8/8/8/8/8/5k2/8/4K2R w - - 0 1"
        (
            piece_squares,
            piece_types,
            piece_colors,
            n_pieces,
            wk,
            bk,
            white_to_move,
        ) = parse_fen_fast(fen)

        assert n_pieces == 1  # Just the rook
        assert wk == 4  # e1
        assert bk == 21  # f3
        assert white_to_move is True

    def test_bare_kings(self):
        """Test parsing position with only kings."""
        fen = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
        *_, n_pieces, wk, bk, _ = parse_fen_fast(fen)

        assert n_pieces == 0
        assert wk == 4  # e1
        assert bk == 36  # e5


class TestHalfKPFeatures:
    """Tests for HalfKP feature extraction."""

    @pytest.mark.parametrize("fen,name", TEST_FENS)
    def test_features_match_reference(self, fen: str, name: str):
        """Test that numba implementation matches reference for various positions."""
        w_new, b_new, stm_new = get_halfkp_features(fen)
        w_ref, b_ref, stm_ref = get_halfkp_features_reference(fen)

        # Sort features since order may differ
        assert sorted(w_new) == sorted(w_ref), f"White features mismatch for {name}"
        assert sorted(b_new) == sorted(b_ref), f"Black features mismatch for {name}"
        assert stm_new == stm_ref, f"STM mismatch for {name}"

    @pytest.mark.parametrize("fen,name", TEST_FENS)
    def test_numpy_version_matches(self, fen: str, name: str):
        """Test that numpy version matches list version."""
        w_list, b_list, stm_list = get_halfkp_features(fen)
        w_np, b_np, stm_np = get_halfkp_features_np(fen)

        assert list(w_np) == w_list, f"White features mismatch for {name}"
        assert list(b_np) == b_list, f"Black features mismatch for {name}"
        assert stm_np == stm_list, f"STM mismatch for {name}"

    def test_feature_count(self):
        """Test that feature count matches piece count."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        w, b, _ = get_halfkp_features(fen)

        # Starting position has 30 non-king pieces
        assert len(w) == 30
        assert len(b) == 30

    def test_feature_bounds(self):
        """Test that all features are within valid HalfKP range."""
        for fen, _ in TEST_FENS:
            w, b, _ = get_halfkp_features(fen)

            for feat in w + b:
                assert 1 <= feat < 41024, f"Feature {feat} out of bounds for {fen}"

    def test_empty_position(self):
        """Test handling of position with only kings."""
        fen = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
        w, b, stm = get_halfkp_features(fen)

        assert w == []
        assert b == []
        assert stm == 0  # White to move

    def test_numpy_empty_position(self):
        """Test numpy version with only kings."""
        fen = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
        w, b, stm = get_halfkp_features_np(fen)

        assert len(w) == 0
        assert len(b) == 0
        assert w.dtype == np.int32
        assert b.dtype == np.int32


class TestThreadSafety:
    """Tests for thread safety of feature extraction."""

    def test_concurrent_extraction(self):
        """Test that concurrent feature extraction works correctly."""
        import concurrent.futures

        fens = [fen for fen, _ in TEST_FENS]
        n_iterations = 100

        def extract_features(fen):
            return fen, get_halfkp_features(fen)

        # Run extractions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(n_iterations):
                for fen in fens:
                    futures.append(executor.submit(extract_features, fen))

            # Verify all results
            for future in futures:
                fen, (w, b, stm) = future.result()
                w_ref, b_ref, stm_ref = get_halfkp_features_reference(fen)

                assert sorted(w) == sorted(w_ref)
                assert sorted(b) == sorted(b_ref)
                assert stm == stm_ref


class TestPerformance:
    """Performance tests (these are informational, not assertions)."""

    @pytest.mark.benchmark
    def test_benchmark_feature_extraction(self):
        """Benchmark feature extraction speed."""
        import time

        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        n_iterations = 10000

        # Warm up JIT
        get_halfkp_features(fen)

        # Benchmark numba version
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            get_halfkp_features(fen)
        t_numba = time.perf_counter() - t0

        # Benchmark reference version
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            get_halfkp_features_reference(fen)
        t_ref = time.perf_counter() - t0

        speedup = t_ref / t_numba
        print(f"\nNumba: {t_numba * 1000:.1f}ms for {n_iterations} iterations")
        print(f"Reference: {t_ref * 1000:.1f}ms for {n_iterations} iterations")
        print(f"Speedup: {speedup:.1f}x")

        # Assert reasonable speedup (should be at least 2x faster)
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"
