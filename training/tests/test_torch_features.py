"""
Tests that the _preprocess torch-native feature extraction produces identical
results to the numpy (raw) path in dummy_chess.
"""

import numpy as np
import pyarrow
import pytest
import torch

import _preprocess
import dummy_chess


TEST_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - - 0 10",
    "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
    "8/5k2/8/8/8/8/6K1/4Q3 w - - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "8/PPPPPPPP/8/8/8/8/pppppppp/8 w - - 0 1",
    "q7/8/8/8/8/8/8/7K w - - 0 1",
    "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3",
]

# Positions that lack both kings produce undefined behavior in HalfKAv2's king-bucket
# indexing.  Indices of such FENs in TEST_FENS (no black king / no kings at all).
_INVALID_FOR_HALFKAV2 = {9, 10}  # "8/PPPPPPPP/..." and "q7/..."


def _make_arrow_buffers(fens: list[str]):
    """Compress FENs and return Arrow buffer pointers."""
    compressed = [dummy_chess.compress_fen(f) for f in fens]
    arr = pyarrow.array(compressed, type=pyarrow.large_binary())
    bufs = arr.buffers()
    return arr, bufs[2].address, bufs[1].address, len(fens)


class TestTorchFeaturesMatchRaw:
    """Verify _preprocess torch-native extraction is identical to dummy_chess raw extraction."""

    @pytest.fixture(autouse=True)
    def setup_buffers(self):
        self.arr, self.data_ptr, self.off_ptr, self.n = _make_arrow_buffers(TEST_FENS)

    @pytest.mark.parametrize("flip", [False, True], ids=["normal", "flipped"])
    def test_halfkp_batch(self, flip: bool):
        raw = dummy_chess.get_halfkp_features_raw(
            self.data_ptr, self.off_ptr, 0, self.n, flip
        )
        tch = _preprocess.get_halfkp_features_torch(
            self.data_ptr, self.off_ptr, 0, self.n, flip
        )
        _assert_tuple_equal(raw, tch, f"HalfKP batch flip={flip}")

    @pytest.mark.parametrize("flip", [False, True], ids=["normal", "flipped"])
    def test_halfkav2_batch(self, flip: bool):
        """Test HalfKAv2 on valid positions only (positions with both kings)."""
        valid_fens = [
            f for i, f in enumerate(TEST_FENS) if i not in _INVALID_FOR_HALFKAV2
        ]
        arr, data_ptr, off_ptr, n = _make_arrow_buffers(valid_fens)
        raw = dummy_chess.get_halfkav2_features_raw(data_ptr, off_ptr, 0, n, flip)
        tch = _preprocess.get_halfkav2_features_torch(data_ptr, off_ptr, 0, n, flip)
        _assert_tuple_equal(raw, tch, f"HalfKAv2 batch flip={flip}")

    @pytest.mark.parametrize("idx", range(len(TEST_FENS)))
    def test_halfkp_single(self, idx: int):
        raw = dummy_chess.get_halfkp_features_raw(self.data_ptr, self.off_ptr, idx, 1)
        tch = _preprocess.get_halfkp_features_torch(self.data_ptr, self.off_ptr, idx, 1)
        _assert_tuple_equal(raw, tch, f"HalfKP single idx={idx}")

    def test_torch_returns_tensors(self):
        tch = _preprocess.get_halfkp_features_torch(
            self.data_ptr, self.off_ptr, 0, self.n
        )
        for i, name in enumerate(["w_idx", "w_off", "b_idx", "b_off", "stm"]):
            assert isinstance(tch[i], torch.Tensor), f"{name} is not a Tensor"
            assert tch[i].dtype == torch.int64, f"{name} dtype is {tch[i].dtype}"

    def test_raw_returns_numpy(self):
        raw = dummy_chess.get_halfkp_features_raw(
            self.data_ptr, self.off_ptr, 0, self.n
        )
        for i, name in enumerate(["w_idx", "w_off", "b_idx", "b_off", "stm"]):
            assert isinstance(raw[i], np.ndarray), f"{name} is not ndarray"

    def test_constants_match(self):
        """Verify _preprocess constants match dummy_chess constants."""
        assert _preprocess.HALFKP_SIZE == dummy_chess.HALFKP_SIZE
        assert _preprocess.HALFKAV2_SIZE == dummy_chess.HALFKAV2_SIZE


def _assert_tuple_equal(raw: tuple, tch: tuple, label: str):
    """Assert all elements of raw (numpy) and tch (torch) tuples are equal."""
    names = ["w_idx", "w_off", "b_idx", "b_off", "stm"]
    assert len(raw) == len(tch) == 5
    for name, r, t in zip(names, raw, tch):
        np.testing.assert_array_equal(r, t.numpy(), err_msg=f"{label}: {name} mismatch")
