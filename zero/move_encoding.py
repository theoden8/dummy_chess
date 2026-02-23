"""AlphaZero move encoding for dummy_chess.

Encodes chess moves as (from_square, move_type) -> policy index.
Total policy size: 64 * 73 = 4672.

Move types (73):
  0-55:  Queen-like moves (sliding + king): 8 directions x 7 distances
  56-63: Knight moves: 8 L-shaped offsets
  64-72: Underpromotions: 3 directions x 3 piece types (knight, bishop, rook)
         Queen promotions are encoded as regular queen moves.

Board layout (matching dummy_chess):
  A1=0, B1=1, ..., H1=7
  A2=8, B2=9, ..., H2=15
  ...
  A8=56, B8=57, ..., H8=63

  file = sq % 8, rank = sq // 8  (rank 0 = rank 1 in chess notation)
"""

import numpy

# --- Constants ---

NUM_SQUARES = 64
NUM_MOVE_TYPES = 73
POLICY_SIZE = NUM_SQUARES * NUM_MOVE_TYPES  # 4672

# 8 queen/king directions: (dfile, drank)
# N, NE, E, SE, S, SW, W, NW
_QUEEN_DIRS = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]

# 8 knight offsets: (dfile, drank)
_KNIGHT_OFFSETS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]

# Underpromotion directions (from white's perspective):
#   left-capture (dfile=-1), forward (dfile=0), right-capture (dfile=+1)
# Pieces: knight=0, bishop=1, rook=2
# Queen promotion is already a queen-move-type, not an underpromotion.
_UNDERPROMO_DFILES = [-1, 0, 1]
_UNDERPROMO_PIECES = [1, 2, 3]  # knight, bishop, rook (matching PIECE enum)

# Promotion piece -> engine encoding (bits 7-6 of to_sq byte)
_PROMO_ENCODE = {
    1: 0 << 6,  # KNIGHT -> PROMOTE_KNIGHT = 0
    2: 1 << 6,  # BISHOP -> PROMOTE_BISHOP = 64
    3: 2 << 6,  # ROOK   -> PROMOTE_ROOK   = 128
    4: 3 << 6,  # QUEEN  -> PROMOTE_QUEEN  = 192
}


def _sq(file: int, rank: int) -> int:
    return file + rank * 8


def _file(sq: int) -> int:
    return sq % 8


def _rank(sq: int) -> int:
    return sq // 8


def _on_board(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def _move_type_queen(direction_idx: int, distance: int) -> int:
    """Queen move type index: direction * 7 + (distance - 1)."""
    return direction_idx * 7 + (distance - 1)


def _move_type_knight(offset_idx: int) -> int:
    """Knight move type index: 56 + offset_idx."""
    return 56 + offset_idx


def _move_type_underpromo(dir_idx: int, piece_idx: int) -> int:
    """Underpromotion move type index: 64 + dir_idx * 3 + piece_idx."""
    return 64 + dir_idx * 3 + piece_idx


# --- Build lookup tables ---


def _build_tables():
    """Build encode/decode lookup tables.

    Returns:
        encode_table: dict mapping (from_sq, to_sq, promo_piece) -> policy_index
            promo_piece is 0 for non-promotions, 1-4 for knight/bishop/rook/queen
        decode_table: numpy array of shape (POLICY_SIZE, 3) -> (from_sq, to_sq, promo_piece)
            -1 for invalid entries
    """
    encode = {}
    decode = numpy.full((POLICY_SIZE, 3), -1, dtype=numpy.int16)

    for from_sq in range(NUM_SQUARES):
        from_file = _file(from_sq)
        from_rank = _rank(from_sq)

        # Queen/king moves: 8 directions x 7 distances
        for dir_idx, (df, dr) in enumerate(_QUEEN_DIRS):
            for dist in range(1, 8):
                to_file = from_file + df * dist
                to_rank = from_rank + dr * dist
                if not _on_board(to_file, to_rank):
                    break
                to_sq = _sq(to_file, to_rank)
                mt = _move_type_queen(dir_idx, dist)
                policy_idx = from_sq * NUM_MOVE_TYPES + mt

                # Check if this is a pawn reaching promotion rank
                # White pawn on rank 6 (0-indexed) moving to rank 7 = queen promo
                # Black pawn on rank 1 moving to rank 0 = queen promo
                is_white_promo = from_rank == 6 and to_rank == 7 and dist == 1
                is_black_promo = from_rank == 1 and to_rank == 0 and dist == 1
                if is_white_promo or is_black_promo:
                    # This slot encodes queen promotion
                    encode[(from_sq, to_sq, 4)] = policy_idx  # QUEEN=4
                    decode[policy_idx] = [from_sq, to_sq, 4]
                else:
                    encode[(from_sq, to_sq, 0)] = policy_idx
                    decode[policy_idx] = [from_sq, to_sq, 0]

        # Knight moves
        for off_idx, (df, dr) in enumerate(_KNIGHT_OFFSETS):
            to_file = from_file + df
            to_rank = from_rank + dr
            if not _on_board(to_file, to_rank):
                continue
            to_sq = _sq(to_file, to_rank)
            mt = _move_type_knight(off_idx)
            policy_idx = from_sq * NUM_MOVE_TYPES + mt
            encode[(from_sq, to_sq, 0)] = policy_idx
            decode[policy_idx] = [from_sq, to_sq, 0]

        # Underpromotions (only from ranks 6->7 for white, 1->0 for black)
        for promo_rank_from, promo_rank_to in [(6, 7), (1, 0)]:
            if from_rank != promo_rank_from:
                continue
            for dir_idx, dfile in enumerate(_UNDERPROMO_DFILES):
                to_file = from_file + dfile
                if not _on_board(to_file, promo_rank_to):
                    continue
                to_sq = _sq(to_file, promo_rank_to)
                for piece_idx, piece in enumerate(_UNDERPROMO_PIECES):
                    mt = _move_type_underpromo(dir_idx, piece_idx)
                    policy_idx = from_sq * NUM_MOVE_TYPES + mt
                    encode[(from_sq, to_sq, piece)] = policy_idx
                    decode[policy_idx] = [from_sq, to_sq, piece]

    return encode, decode


_ENCODE_TABLE, DECODE_TABLE = _build_tables()


def encode_move(from_sq: int, to_sq: int, promo_piece: int = 0) -> int:
    """Encode a move as a policy index.

    Args:
        from_sq: source square (0-63)
        to_sq: destination square (0-63, without promotion bits)
        promo_piece: 0 for normal moves, 1=knight, 2=bishop, 3=rook, 4=queen

    Returns:
        policy index (0-4671)
    """
    return _ENCODE_TABLE[(from_sq, to_sq, promo_piece)]


def decode_move(policy_idx: int) -> tuple[int, int, int]:
    """Decode a policy index to (from_sq, to_sq, promo_piece).

    Returns:
        (from_sq, to_sq, promo_piece) where promo_piece is 0 for non-promotions
    """
    row = DECODE_TABLE[policy_idx]
    return int(row[0]), int(row[1]), int(row[2])


def encode_move_t(move: int) -> int:
    """Encode from engine move_t format.

    move_t is uint16: (from_sq << 8) | to_byte
    where to_byte has bits 5-0 = to_sq, bits 7-6 = promotion type.
    Promotion bits: 0=knight, 64=bishop, 128=rook, 192=queen.
    """
    from_sq = (move >> 8) & 0x3F
    to_sq = move & 0x3F
    promo_bits = move & 0xC0  # bits 7-6 of low byte

    # Determine if this is a promotion: pawn reaching rank 0 or 7
    to_rank = _rank(to_sq)
    from_rank = _rank(from_sq)
    is_promo = (to_rank == 7 and from_rank == 6) or (to_rank == 0 and from_rank == 1)

    if is_promo:
        # Map engine promo bits to piece index
        promo_map = {0: 1, 64: 2, 128: 3, 192: 4}  # knight, bishop, rook, queen
        promo_piece = promo_map[promo_bits]
    else:
        promo_piece = 0

    return _ENCODE_TABLE[(from_sq, to_sq, promo_piece)]


def decode_to_move_t(policy_idx: int) -> int:
    """Decode policy index to engine move_t format.

    Returns:
        move_t: (from_sq << 8) | to_byte
    """
    from_sq, to_sq, promo_piece = decode_move(policy_idx)
    if promo_piece > 0:
        to_byte = to_sq | _PROMO_ENCODE[promo_piece]
    else:
        to_byte = to_sq
    return (from_sq << 8) | to_byte


# --- Precomputed numpy arrays for batch operations ---

# Encode table as numpy array: shape (64, 64, 5) -> policy_index
# Axes: [from_sq, to_sq, promo_piece], value = policy_idx or -1
ENCODE_ARRAY = numpy.full((64, 64, 5), -1, dtype=numpy.int16)
for (fsq, tsq, pp), pidx in _ENCODE_TABLE.items():
    ENCODE_ARRAY[fsq, tsq, pp] = pidx
