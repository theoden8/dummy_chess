#!/usr/bin/env python3
"""
Test C++ PGN parser against python-chess for correctness.
"""

import io
import pathlib
import re
import sys

import chess.pgn
import dummy_chess

# Path to PGN fixtures directory
FIXTURES_DIR = pathlib.Path(__file__).parent / "pgn_fixtures"


def load_fixture(name: str) -> str:
    """Load a PGN fixture file."""
    return (FIXTURES_DIR / name).read_text()


def test_variant_detection():
    """Test that variant detection works correctly."""
    print("Testing variant detection...")

    # Standard chess
    standard_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    compressed = dummy_chess.compress_fen(standard_fen)
    variant = dummy_chess.get_variant(compressed)
    assert variant == "standard", f"Expected 'standard', got '{variant}'"
    print(f"  Standard: PASS")

    # Test batch variant detection
    fens = [standard_fen, standard_fen]
    compressed_batch = dummy_chess.compress_fens_batch(fens)
    variants = dummy_chess.get_variant_batch(compressed_batch)
    assert variants == ["standard", "standard"], (
        f"Expected ['standard', 'standard'], got {variants}"
    )
    print(f"  Batch: PASS")

    print("Variant detection: PASS")
    print()


def test_ply_position_correspondence():
    """Test that ply index correctly corresponds to the board position."""
    print("Testing ply/position correspondence...")

    # Game where we know exactly what position should be at each ply
    pgn = """1. e4 { [%eval 0.3] } 1... e5 { [%eval 0.2] } 2. Nf3 { [%eval 0.4] } 2... Nc6 { [%eval 0.3] } 1-0"""

    results = dummy_chess.parse_pgn_with_evals(pgn)

    # Expected positions (before each move with eval)
    expected = [
        (
            0,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        ),  # Starting position before 1. e4
        (
            1,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
        ),  # After 1. e4, before 1... e5
        (
            2,
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",
        ),  # After 1... e5, before 2. Nf3
        (
            3,
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
        ),  # After 2. Nf3, before 2... Nc6
    ]

    if len(results) != len(expected):
        print(f"  FAIL: Expected {len(expected)} positions, got {len(results)}")
        return False

    for (exp_ply, exp_board), (compressed_fen, score, ply) in zip(expected, results):
        fen = dummy_chess.decompress_fen(compressed_fen)
        board_part = fen.split()[0]  # Just the board part

        if ply != exp_ply:
            print(f"  FAIL: Expected ply {exp_ply}, got {ply}")
            return False

        if board_part != exp_board:
            print(f"  FAIL at ply {ply}:")
            print(f"    Expected: {exp_board}")
            print(f"    Got:      {board_part}")
            return False

    print(f"  All {len(expected)} ply/position checks: PASS")
    print()
    return True


def test_comments_preserved():
    """Test that comments with evals are correctly extracted."""
    print("Testing comment/eval extraction...")

    # Test various eval formats
    test_cases = [
        # (pgn, expected_evals)
        ("1. e4 { [%eval 0.30] } 1... e5 { [%eval -0.10] } 1-0", [30, -10]),
        ("1. e4 { [%eval 1.5] } 1... e5 { [%eval 2.75] } 1-0", [150, 275]),
        ("1. e4 { [%eval #5] } 1... e5 { [%eval #-3] } 1-0", [10000, -10000]),
        ("1. e4 { [%eval 0.0] } 1-0", [0]),
        ("1. e4 { [%eval -0.00] } 1-0", [0]),
        # With clock annotations
        (
            "1. e4 { [%eval 0.25] [%clk 0:03:00] } 1... e5 { [%clk 0:02:58] [%eval 0.30] } 1-0",
            [25, 30],
        ),
        # Whitespace variations
        ("1. e4 { [%eval  0.5 ] } 1-0", [50]),
        # No eval in comment
        ("1. e4 { This is a comment without eval } 1... e5 { [%eval 0.2] } 1-0", [20]),
    ]

    for pgn, expected_evals in test_cases:
        results = dummy_chess.parse_pgn_with_evals(pgn)
        actual_evals = [score for _, score, _ in results]

        if actual_evals != expected_evals:
            print(f"  FAIL: '{pgn[:50]}...'")
            print(f"    Expected: {expected_evals}")
            print(f"    Got: {actual_evals}")
            return False

    print(f"  All {len(test_cases)} eval extraction tests: PASS")
    print()
    return True


def get_regression_pgns() -> list[str]:
    """Load regression test PGNs from fixtures."""
    return [
        load_fixture("regression_enpassant_horizontal_pin.pgn"),
        load_fixture("regression_re1e4_disambiguation.pgn"),
        load_fixture("regression_50move_checkmate.pgn"),
    ]


def get_test_pgns() -> list[str]:
    """Load test PGNs from fixtures."""
    return [
        load_fixture("test_basic_evals.pgn"),
        load_fixture("test_mate_scores.pgn"),
        load_fixture("test_clock_annotations.pgn"),
        load_fixture("test_variations.pgn"),
        load_fixture("test_nags.pgn"),
        load_fixture("test_from_position.pgn"),
        load_fixture("test_negative_evals.pgn"),
        load_fixture("test_precise_evals.pgn"),
    ]


def parse_with_python_chess(pgn_text: str) -> list[tuple[str, int, int]]:
    """Parse PGN with python-chess and extract (fen, score_cp, ply) tuples."""
    results = []
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return results

    board = game.board()
    ply = 0

    for node in game.mainline():
        comment = node.comment or ""

        # Parse [%eval X] from comment
        match = re.search(r"\[%eval\s+([^\]]+)\]", comment)
        if match:
            eval_str = match.group(1).strip()
            score_cp = None

            if eval_str.startswith("#"):
                try:
                    mate_in = int(eval_str[1:])
                    score_cp = 10000 if mate_in > 0 else -10000
                except ValueError:
                    pass
            else:
                try:
                    score_cp = int(float(eval_str) * 100)
                except ValueError:
                    pass

            if score_cp is not None:
                results.append((board.fen(), score_cp, ply))

        board.push(node.move)
        ply += 1

    return results


def parse_with_cpp(pgn_text: str) -> list[tuple[str, int, int]]:
    """Parse PGN with C++ parser and extract (fen, score_cp, ply) tuples."""
    results = []
    for compressed_fen, score_cp, ply in dummy_chess.parse_pgn_with_evals(pgn_text):
        fen = dummy_chess.decompress_fen(compressed_fen)
        results.append((fen, score_cp, ply))
    return results


def fen_equal(fen1: str, fen2: str) -> bool:
    """Compare FENs, ignoring move counters and en passant square.

    En passant handling differs between engines - some clear it after
    one ply if no pawn can capture, others keep it. Both are valid.
    """
    # Split and compare first 3 parts (board, turn, castling)
    # Skip en passant (part 4) and move counters (parts 5-6)
    parts1 = fen1.split()[:3]
    parts2 = fen2.split()[:3]
    return parts1 == parts2


def test_pgn_parser():
    """Compare C++ and Python parsers on test cases."""
    print("Testing C++ PGN parser against python-chess...")
    print()

    all_passed = True

    for i, pgn_text in enumerate(get_test_pgns()):
        # Get event name for display
        event = "Unknown"
        for line in pgn_text.split("\n"):
            if line.startswith('[Event "'):
                event = line.split('"')[1]
                break

        print(f"Test {i + 1}: {event}")

        python_results = parse_with_python_chess(pgn_text)
        cpp_results = parse_with_cpp(pgn_text)

        if len(python_results) != len(cpp_results):
            print(
                f"  FAIL: Position count mismatch: Python={len(python_results)}, C++={len(cpp_results)}"
            )
            all_passed = False
            continue

        test_passed = True
        for j, (py_res, cpp_res) in enumerate(zip(python_results, cpp_results)):
            py_fen, py_score, py_ply = py_res
            cpp_fen, cpp_score, cpp_ply = cpp_res

            if py_ply != cpp_ply:
                print(
                    f"  FAIL: Position {j}: ply mismatch: Python={py_ply}, C++={cpp_ply}"
                )
                test_passed = False
                continue

            if py_score != cpp_score:
                print(
                    f"  FAIL: Position {j}: score mismatch: Python={py_score}, C++={cpp_score}"
                )
                test_passed = False
                continue

            if not fen_equal(py_fen, cpp_fen):
                print(f"  FAIL: Position {j}: FEN mismatch:")
                print(f"    Python: {py_fen}")
                print(f"    C++:    {cpp_fen}")
                test_passed = False
                continue

        if test_passed:
            print(f"  PASS: {len(python_results)} positions match")
        else:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


def test_regression_pgns():
    """Test PGNs that previously caused parser failures."""
    print("Testing regression PGNs...")
    print()

    all_passed = True

    for i, pgn_text in enumerate(get_regression_pgns()):
        # Get event/site for display
        event = "Unknown"
        site = ""
        for line in pgn_text.split("\n"):
            if line.startswith('[Event "'):
                event = line.split('"')[1]
            if line.startswith('[Site "'):
                site = line.split('"')[1]

        print(f"Regression {i + 1}: {event}")
        if site:
            print(f"  Site: {site}")

        try:
            # Just verify it parses without crashing
            results = dummy_chess.parse_pgn_with_evals(pgn_text)
            print(f"  PASS: Parsed successfully ({len(results)} positions with evals)")
        except Exception as e:
            print(f"  FAIL: {e}")
            all_passed = False

    print()
    return all_passed


if __name__ == "__main__":
    failed = False

    # Run all tests
    test_variant_detection()

    if not test_ply_position_correspondence():
        failed = True

    if not test_comments_preserved():
        failed = True

    if test_pgn_parser() != 0:
        failed = True

    if not test_regression_pgns():
        failed = True

    sys.exit(1 if failed else 0)
