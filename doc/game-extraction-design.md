# Game Position Extraction Design

Extract random positions from Lichess game archives for NNUE training data.

## Data Source

Lichess provides monthly game archives at https://database.lichess.org/

- **Game list**: https://database.lichess.org/standard/list.txt
- **Format**: PGN files compressed with zstd (`.pgn.zst`)
- **Size**: ~50-100 GB per month compressed, billions of games total

### Variants

| Variant | URL Pattern | Notes |
|---------|-------------|-------|
| Standard | `/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst` | Main target |
| Chess960 | `/chess960/lichess_db_chess960_rated_YYYY-MM.pgn.zst` | Use for Chess960-specific training |
| Crazyhouse | `/crazyhouse/...` | Different piece placement rules, separate model |

### Known Issues (from Lichess)

| Date Range | Issue | Action |
|------------|-------|--------|
| November 2023 | Chess960 rematches have invalid castling rights | Skip Chess960 for 2023-11 |
| March 12, 2021 | Incorrect results (datacenter fire) | Skip 2021-03 entirely |
| Feb 9, 2021 | Resignations after game end | Filter by result consistency |
| Dec 2020 - Jan 2021 | Variant games analyzed with wrong NNUE | Skip variants for these months |
| July 31 - Aug 16, 2020 | Incorrect opening evals (up to 15 plies) | We skip evals anyway, safe |
| December 2016 (esp. 9th) | Many incorrect evaluations | We skip evals anyway, safe |
| Before 2016 | Mate distances may be incorrect | We skip evals anyway, safe |
| June 2020, before March 2016 | Self-play in rated games | Minor, ignore |
| Up to August 2016 | 7 illegal castling moves | Negligible |

**Blocklist** (skip entirely):
- `2021-03` (datacenter fire, bad results)

**Recommendation**: We extract positions WITHOUT existing evals and run fresh Stockfish analysis, so most eval-related issues don't affect us. The March 2021 datacenter issue is the only one requiring a full skip.

## Extraction Strategy

### Why Not Use Existing Evals?

~6% of Lichess games have `[%eval]` annotations, but:
1. These are already in `lichess_db_eval.jsonl.zst` (the evals database)
2. Some months have known evaluation bugs
3. We want diverse positions, not just analyzed ones

### Our Approach

For each game:
1. Parse PGN, play through moves
2. Select ONE random position (avoids correlated samples from same game)
3. **Skip** positions that already have `[%eval]` annotation
4. **Skip** positions in opening (< N plies) or near game end
5. Run Stockfish to get evaluation
6. Output (compressed_fen, score) to parquet

### Position Selection Criteria

```python
def should_extract(game, ply_index, position) -> bool:
    # Skip if position already has eval (it's in the evals database)
    if has_eval_annotation(position):
        return False
    
    # Skip opening positions (too book-dependent)
    if ply_index < MIN_PLY:  # e.g., 10
        return False
    
    # Skip positions near game end (often trivial)
    if ply_index > total_plies - END_MARGIN:  # e.g., 5 plies from end
        return False
    
    # Skip positions where game is decided (checkmate, stalemate coming)
    if is_game_over(position):
        return False
    
    # Skip positions with very few pieces (use endgame tablebases instead)
    if piece_count(position) <= TB_PIECES:  # e.g., 6
        return False
    
    return True
```

### Filtering by Game Quality

```python
def should_process_game(game) -> bool:
    # Skip bot games
    if is_bot_game(game):
        return False
    
    # Optional: filter by rating
    if min_elo and (white_elo < min_elo or black_elo < min_elo):
        return False
    
    # Skip games with too few moves
    if total_plies < MIN_GAME_LENGTH:
        return False
    
    # Skip abandoned games
    if is_abandoned(game):
        return False
    
    return True
```

## Implementation

### CLI Interface

```bash
# Download and process a single month
uv run python preprocess.py games \
    --month 2024-01 \
    --engine /path/to/stockfish \
    --depth 12 \
    --output data/games_2024_01.parquet \
    --max 1000000

# Process multiple months
uv run python preprocess.py games \
    --months 2023-01 2023-02 2023-03 \
    --engine /path/to/stockfish \
    --output data/games_2023_q1.parquet

# Resume interrupted processing
uv run python preprocess.py games \
    --month 2024-01 \
    --resume \
    --output data/games_2024_01.parquet
```

### Resume Support

Track progress via:
1. **Game index checkpoint**: Store last processed game index in sidecar file
2. **Parquet append**: Write in batches, append to existing file
3. **Download resume**: HTTP Range headers for partial downloads

Checkpoint file format (`.progress.json`):
```json
{
    "month": "2024-01",
    "variant": "standard",
    "games_processed": 1234567,
    "positions_extracted": 987654,
    "bytes_downloaded": 12345678901,
    "last_game_offset": 98765432
}
```

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Process                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   HTTP Download ──► Zstd Decompress ──► PGN Parser              │
│        │                                     │                   │
│        │ (streaming)                         │ (streaming)       │
│        ▼                                     ▼                   │
│   [chunk buffer]                        Game Queue              │
│                                              │                   │
│                              ┌───────────────┼───────────────┐   │
│                              ▼               ▼               ▼   │
│                         [Worker 1]     [Worker 2]     [Worker N] │
│                         Stockfish      Stockfish      Stockfish  │
│                              │               │               │   │
│                              └───────────────┼───────────────┘   │
│                                              ▼                   │
│                                      Parquet Writer             │
│                                     (batch append)               │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Efficiency

- **Streaming download**: Don't buffer entire file
- **Streaming decompression**: zstd streaming reader
- **Streaming PGN parsing**: One game at a time
- **Batch writing**: Accumulate N positions, write as row group

### Stockfish Integration

Reuse existing `EnginePool` pattern from puzzles processing:

```python
class EnginePool:
    """Pool of Stockfish processes for parallel evaluation."""
    
    def __init__(self, engine_path: str, num_workers: int, depth: int):
        self.engines = [
            chess.engine.SimpleEngine.popen_uci(engine_path)
            for _ in range(num_workers)
        ]
        self.depth = depth
    
    def evaluate(self, fen: str) -> int:
        """Get centipawn evaluation for position."""
        engine = self.get_available_engine()
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].relative.score(mate_score=10000)
        return score
```

## Output Format

Same as other preprocessing: parquet with compressed FEN and score.

```
Columns:
- fen: bytes (compressed FEN)
- score: int32 (centipawns from white's perspective)
- ply: int16 (optional: position's ply in game, for analysis)
- elo_avg: int16 (optional: average of player ratings)
```

## Estimated Yields

| Month Size | Games | Extractable Positions | After Filtering |
|------------|-------|----------------------|-----------------|
| 50 GB compressed | ~80M games | ~80M (1 per game) | ~60M (quality filters) |

With 10 months of data: ~600M unique positions from real games.

## Chess960 Considerations

For Chess960:
- Parse starting position from FEN tag
- Validate castling rights (skip Nov 2023 corrupt games)
- Same extraction logic, but store variant tag

```python
if variant == "chess960":
    # Validate starting FEN castling rights
    if not validate_chess960_castling(starting_fen):
        skip_game()
```

## Configuration

```python
@dataclasses.dataclass
class GameExtractionConfig:
    # Position selection
    min_ply: int = 10              # Skip opening
    end_margin: int = 5            # Skip near game end
    min_pieces: int = 7            # Skip simple endgames
    
    # Game filtering
    min_elo: int | None = None     # Optional rating floor
    min_game_length: int = 20      # Minimum plies
    skip_bots: bool = True         # Skip bot games
    
    # Engine settings
    depth: int = 12                # Stockfish depth
    num_workers: int = 4           # Parallel engines
    
    # Output
    batch_size: int = 10000        # Positions per parquet row group
    
    # Variants
    variants: list[str] = field(default_factory=lambda: ["standard"])
```

## TODO

- [ ] Implement PGN streaming parser with eval annotation detection
- [ ] Add HTTP streaming download with resume
- [ ] Implement position extraction with Stockfish pool
- [ ] Add progress checkpointing for resume
- [ ] Support Chess960 variant
- [ ] Add rating-based stratified sampling option
