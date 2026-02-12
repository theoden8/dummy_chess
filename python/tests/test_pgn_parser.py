#!/usr/bin/env python3
"""
Test C++ PGN parser against python-chess for correctness.
"""

import chess.pgn
import io
import dummy_chess


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


# Regression test PGNs - games that previously caused parser failures
REGRESSION_PGNS = [
    # Game with exd3 en passant that was incorrectly blocked by horizontal pin check
    # The horizontal pin check didn't verify king was on the same rank as the pawns,
    # so it incorrectly blocked a valid vertical-file en passant capture
    """[Event "Rated Classical game"]
[Site "https://lichess.org/wn8uhw1u"]
[White "ryryry"]
[Black "Magucho"]
[Result "1-0"]
[UTCDate "2013.06.29"]
[UTCTime "06:50:12"]
[WhiteElo "1781"]
[BlackElo "1722"]
[WhiteRatingDiff "+9"]
[BlackRatingDiff "-9"]
[ECO "C31"]
[Opening "King's Gambit, Falkbeer Countergambit, Blackburne Attack"]
[TimeControl "600+0"]
[Termination "Normal"]

1. e4 e5 2. f4 d5 3. Nf3 dxe4 4. Nxe5 Nf6 5. Bc4 Qe7 6. Bxf7+ Kd8 7. Bb3 Be6 8. O-O Bxb3 9. axb3 Nd5 10. c4 Nxf4 11. Rxf4 Qxe5 12. d4 exd3 13. Qxd3+ Qd6 14. Rd4 1-0""",
    # Game with Re1e4 - full square disambiguation (triggered ply.back() != s assertion)
    # The parser generates "R1e4" but input was "Re1e4"
    # Also has 3-fold repetition before final checkmate (moves 62-67)
    """[Event "Rated Blitz game"]
[Site "https://lichess.org/2cj9hphe"]
[White "tiggran"]
[Black "barriosgb2"]
[Result "1-0"]
[UTCDate "2012.12.31"]
[UTCTime "23:08:52"]
[WhiteElo "1533"]
[BlackElo "1712"]
[ECO "B10"]
[Opening "Caro-Kann Defense"]
[TimeControl "300+0"]
[Termination "Normal"]

1. e4 c6 2. Nf3 d5 3. exd5 cxd5 4. d4 Nc6 5. Nc3 e6 6. Be2 Bd6 7. O-O Nf6 8. Bg5 O-O 9. Qd2 Be7 10. Rfe1 Qc7 11. Bf4 Qd7 12. Bb5 a6 13. Ba4 b5 14. Bb3 Qa7 15. Nh4 Qd7 16. Bh6 gxh6 17. Qxh6 Ne4 18. Nxe4 dxe4 19. Rxe4 f5 20. Rxe6 Kh8 21. Nxf5 Rxf5 22. Rae1 Bf8 23. Qh3 Rg5 24. Re1e4 Qg7 25. g3 Bxe6 26. Bxe6 Re8 27. d5 Ne5 28. f4 Nf3+ 29. Kh1 Rg6 30. f5 Rg5 31. Kg2 Nd4 32. c3 Nxe6 33. dxe6 Bd6 34. g4 Qf6 35. Qd3 Bb8 36. Qd7 Qh6 37. e7 Qxh2+ 38. Kf3 Qg3+ 39. Ke2 Qg2+ 40. Kd1 Qxe4 41. Qxe8+ Rg8 42. Qf7 Qxg4+ 43. Kc2 Qg2+ 44. Kb3 a5 45. e8=Q a4+ 46. Ka3 Bd6+ 47. b4 axb3+ 48. Kxb3 Rxe8 49. Qxe8+ Kg7 50. Qd7+ Kh6 51. Qxd6+ Kg5 52. f6 Qf3 53. Qe5+ Kg6 54. Qxb5 Qxf6 55. a4 Qe6+ 56. Ka3 Qd6+ 57. Qb4 Qd3 58. a5 h5 59. Qb6+ Kg5 60. Qc5+ Kg4 61. Kb4 Qb1+ 62. Kc4 Qa2+ 63. Kd3 Qb1+ 64. Kc4 Qa2+ 65. Kd4 Qd2+ 66. Kc4 Qa2+ 67. Kb5 Qb3+ 68. Qb4+ Qxb4+ 69. Kxb4 h4 70. a6 h3 71. a7 h2 72. a8=Q h1=B 73. Qxh1 Kf4 74. c4 Kg3 75. Qe1+ Kf3 76. Qd2 Ke4 77. Qc3 Kf4 78. Qd3 Ke5 79. c5 Ke6 80. c6 Ke5 81. c7 Ke6 82. Qd4 Ke7 83. c8=Q Kf7 84. Qd6 Kg7 85. Qcc7+ Kg8 86. Qdd8# 1-0""",
    # Game with checkmate on the 50th move (100 half-moves) - checkmate takes precedence over 50-move draw
    """[Event "Rated Bullet game"]
[Site "https://lichess.org/49iz7ttf"]
[White "silverbrutus"]
[Black "458"]
[Result "0-1"]
[UTCDate "2013.04.18"]
[UTCTime "01:16:15"]
[WhiteElo "1569"]
[BlackElo "1545"]
[WhiteRatingDiff "-13"]
[BlackRatingDiff "+13"]
[ECO "B06"]
[Opening "Modern Defense"]
[TimeControl "0+1"]
[Termination "Normal"]

1. e4 g6 2. f4 b6 3. Nf3 Bb7 4. Bc4 Bg7 5. Nc3 e6 6. d4 a6 7. O-O b5 8. Bb3 b4 9. Ne2 a5 10. e5 a4 11. Nh4 axb3 12. cxb3 g5 13. fxg5 f6 14. gxf6 Bxf6 15. exf6 Qxf6 16. Ng3 Qxh4 17. Bh6 Qxh6 18. Rf3 Bxf3 19. Qxf3 Qf6 20. Rf1 Qxf3 21. Rxf3 Nc6 22. Ne4 Nxd4 23. Nf6+ Nxf6 24. Rxf6 Ke7 25. Rh6 Rhg8 26. Kf2 Raf8+ 27. Ke3 Nf5+ 28. Kd3 Nxh6 29. Kc4 d5+ 30. Kxb4 Rxg2 31. Kb5 Rxb2 32. a4 Rxb3+ 33. Kc6 Rh3 34. Kxc7 Rxh2 35. a5 Ra2 36. a6 e5 37. a7 d4 38. a8=Q Rfxa8 39. Kc6 e4 40. Kc5 d3 41. Kc4 e3 42. Kd4 Rd8+ 43. Kxe3 d2 44. Kf4 d1=Q 45. Kg5 Qh1 46. Kf4 Rg2 47. Kf3 Rf8+ 48. Ke3 Kd8 49. Ke4 Kc8 50. Kd4 Kb7 51. Kd5 Ka8 52. Ke5 Ra2 53. Ke6 Ra1 54. Kd6 Ng4 55. Ke6 h5 56. Ke7 h4 57. Kd6 h3 58. Kc5 h2 59. Kd4 Qg1+ 60. Kc4 h1=Q 61. Kd3 Qh8 62. Ke4 Kb7 63. Kd5 Qgh1+ 64. Kc4 Rfa8 65. Kc5 Nf2 66. Kb4 Nd1 67. Kc4 Kc7 68. Kc5 Kd7 69. Kb5 Ke7 70. Kb4 Kf6 71. Kc4 Ke6 72. Kd3 Kf5 73. Kc4 Kg4 74. Kb4 Kf3 75. Kb5 Kg2 76. Kc6 Kf1+ 77. Kd6 Kg1 78. Ke6 Kg2 79. Kf5 Kh2 80. Ke6 Kh3 81. Ke7 Kh4 82. Kd7 Kh5 83. Kd6 Kh6 84. Kc5 Kh7 85. Kc4 Kg8 86. Kc5 Kf8 87. Kc4 Ke8 88. Kc5 Kd8 89. Kc4 Kc8 90. Kc5 Kb8 91. Kc4 Ka7 92. Kc5 Ka6 93. Kc4 Ka5 94. Kc5 Ka4 95. Kc4 Ka3 96. Kc5 Ka2 97. Kc4 Kb1 98. Kc5 Kc1 99. Kc4 Nb2+ 100. Kc5 Kd1 101. Kb6 Ke1 102. Kc5 Kf1 103. Kb5 Kg1 104. Kc5 Kg2 105. Kb5 Nd1 106. Kc5 Rc1+ 107. Kb5 Kf3 108. Kb4 Ke4 109. Kb5 Kd3 110. Kb4 Qb8# 0-1""",
]


# Test PGN games with various features
TEST_PGNS = [
    # Basic game with evals
    """[Event "Test"]
[Result "1-0"]

1. e4 { [%eval 0.3] } 1... e5 { [%eval 0.25] } 2. Nf3 { [%eval 0.35] } 2... Nc6 { [%eval 0.3] } 1-0""",
    # Game with mate scores
    """[Event "Mate Test"]
[Result "1-0"]

1. e4 { [%eval 0.3] } 1... e5 { [%eval #-10] } 2. Qh5 { [%eval #3] } 1-0""",
    # Game with clock annotations
    """[Event "Blitz"]
[Result "1-0"]

1. d4 { [%eval 0.2] [%clk 0:03:00] } 1... d5 { [%eval 0.3] [%clk 0:02:58] } 2. c4 { [%eval 0.25] [%clk 0:02:55] } 1-0""",
    # Game with variations (should be skipped)
    """[Event "Variations"]
[Result "1-0"]

1. e4 { [%eval 0.3] } 1... e5 (1... c5 { [%eval 0.4] }) { [%eval 0.25] } 2. Nf3 { [%eval 0.35] } 1-0""",
    # Game with NAGs
    """[Event "NAGs"]
[Result "1-0"]

1. e4! { [%eval 0.3] } 1... e5?! { [%eval 0.4] } 2. Nf3!! { [%eval 0.35] } 1-0""",
    # Chess960 / FEN start position
    """[Event "From Position"]
[FEN "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]
[Result "1-0"]

3. Bb5 { [%eval 0.5] } 3... a6 { [%eval 0.45] } 1-0""",
    # Negative evals
    """[Event "Black Advantage"]
[Result "0-1"]

1. e4 { [%eval -0.5] } 1... e5 { [%eval -0.75] } 0-1""",
    # Deep eval decimals
    """[Event "Precise"]
[Result "1/2-1/2"]

1. e4 { [%eval 0.17] } 1... e5 { [%eval 0.23] } 1/2-1/2""",
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
        import re

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

    for i, pgn_text in enumerate(TEST_PGNS):
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

    for i, pgn_text in enumerate(REGRESSION_PGNS):
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
    import sys

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
