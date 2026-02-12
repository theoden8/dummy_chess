#!/usr/bin/env python3
"""
Download Lichess training data (puzzles, evaluations, and game archives).

Style guide: No 'from X import Y' except for typing. Acceptable aliases: np, pl, plt, sp, sns, tqdm.

Usage:
    uv run python download.py puzzles
    uv run python download.py evals
    uv run python download.py games --list                    # List available months
    uv run python download.py games --month 2024-01           # Download one month
    uv run python download.py games --all                     # Download all months
    uv run python download.py games --after 2023-01           # Download from 2023-01 onwards
    uv run python download.py games --process -e stockfish    # Download and process

The script supports:
- Progress bar with download speed
- Resume interrupted downloads
- Integrity verification via SHA256 hash
- Batch downloading of game archives
"""

import argparse
import hashlib
import io
import pathlib
import re
import socket
import subprocess
import sys
import tempfile
import time
import typing
import urllib.request
import urllib.error

import tqdm.auto as tqdm

# Retry configuration
MAX_RETRIES = 10
RETRY_DELAY_BASE = 5  # seconds, doubles each retry (exponential backoff)
RETRY_DELAY_MAX = 300  # max 5 minutes between retries

# Exceptions that are worth retrying (transient network issues)
RETRYABLE_EXCEPTIONS = (
    urllib.error.URLError,  # Network unreachable, DNS failure, etc.
    socket.timeout,  # Connection timeout
    ConnectionResetError,  # Connection reset by peer
    ConnectionRefusedError,  # Server temporarily refusing connections
    BrokenPipeError,  # Connection dropped
    OSError,  # Various network-related OS errors
)

# Lichess database URLs
DATASETS = {
    "puzzles": {
        "url": "https://database.lichess.org/lichess_db_puzzle.csv.zst",
        "filename": "lichess_db_puzzle.csv.zst",
        "description": "Lichess puzzle database (~287 MB)",
    },
    "evals": {
        "url": "https://database.lichess.org/lichess_db_eval.jsonl.zst",
        "filename": "lichess_db_eval.jsonl.zst",
        "description": "Lichess evaluation database (~2.5 GB)",
    },
}

# Game archive URLs
GAMES_LIST_URL = "https://database.lichess.org/{variant}/list.txt"
GAMES_HASH_URL = "https://database.lichess.org/{variant}/sha256sums.txt"

import libtorrent as lt


class TorrentStream(io.RawIOBase):
    """
    A file-like object that streams data from a BitTorrent download.

    Uses sequential download mode to ensure pieces are downloaded in order,
    allowing the file to be read as it downloads. Prefetches ahead to keep
    the download saturated while processing.
    """

    # Number of pieces to prefetch ahead (1 MB pieces typical)
    # Reduced from 64 to 32 to save memory (~32 MB prefetch buffer)
    PREFETCH_PIECES = 32

    def __init__(
        self,
        torrent_url: str,
        save_dir: pathlib.Path | None = None,
        max_download_rate: int = 0,
        max_upload_rate: int = 100 * 1024,
        prefetch_pieces: int = 64,
    ):
        """
        Initialize torrent stream.

        Args:
            torrent_url: URL to .torrent file
            save_dir: Directory to save the file (default: temp directory)
            max_download_rate: Max download speed in bytes/sec (0 = unlimited)
            max_upload_rate: Max upload speed in bytes/sec
            prefetch_pieces: Number of pieces to prefetch ahead (default 64 = ~64 MB)
        """
        super().__init__()
        self._torrent_url = torrent_url
        self._position = 0
        self._closed = False
        self._prefetch_pieces = prefetch_pieces

        # Fetch torrent file
        with urllib.request.urlopen(torrent_url, timeout=30) as resp:
            torrent_data = resp.read()

        # Parse torrent
        self._info = lt.torrent_info(lt.bdecode(torrent_data))
        self._total_size = self._info.total_size()
        self._piece_length = self._info.piece_length()
        self._num_pieces = self._info.num_pieces()

        # Create session with memory-efficient settings
        self._session = lt.session()
        settings = {
            "download_rate_limit": max_download_rate,
            "upload_rate_limit": max_upload_rate,
            "connections_limit": 50,  # Reduced from 200 to save memory
            "active_downloads": 1,
            "active_seeds": 1,
        }
        self._session.apply_settings(settings)

        # Set up save directory
        if save_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self._save_dir = pathlib.Path(self._temp_dir.name)
        else:
            self._temp_dir = None
            self._save_dir = save_dir
            self._save_dir.mkdir(parents=True, exist_ok=True)

        # Add torrent with sequential download
        params = lt.add_torrent_params()
        params.ti = self._info
        params.save_path = str(self._save_dir)
        params.flags |= lt.torrent_flags.sequential_download

        self._handle = self._session.add_torrent(params)
        self._file_path = self._save_dir / self._info.name()

        # Wait for metadata and file allocation
        while not self._handle.status().has_metadata:
            time.sleep(0.1)

    @property
    def name(self) -> str:
        return self._info.name()

    @property
    def total_size(self) -> int:
        return self._total_size

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False  # Sequential only for streaming

    def _wait_for_piece(self, piece_index: int, timeout: float = 300) -> bool:
        """Wait for a specific piece to be downloaded."""
        start = time.monotonic()
        while True:
            if self._handle.have_piece(piece_index):
                return True
            if time.monotonic() - start > timeout:
                return False
            # Prioritize this piece and nearby pieces
            self._handle.piece_priority(piece_index, 7)  # Highest priority
            time.sleep(0.1)

    def _wait_for_bytes(self, start: int, end: int, timeout: float = 300) -> bool:
        """Wait for a byte range to be available and prefetch ahead."""
        start_piece = start // self._piece_length
        end_piece = min((end - 1) // self._piece_length, self._num_pieces - 1)

        # Prefetch: set high priority for pieces ahead of current read
        prefetch_end = min(end_piece + self._prefetch_pieces, self._num_pieces - 1)
        for piece_idx in range(end_piece + 1, prefetch_end + 1):
            if not self._handle.have_piece(piece_idx):
                self._handle.piece_priority(piece_idx, 6)  # High priority for prefetch

        # Wait for the pieces we actually need now
        for piece_idx in range(start_piece, end_piece + 1):
            if not self._wait_for_piece(piece_idx, timeout):
                return False
        return True

    def readinto(self, b: typing.Any) -> int:
        """Read bytes into a pre-allocated buffer."""
        if self._closed:
            raise ValueError("I/O operation on closed file")

        if self._position >= self._total_size:
            return 0

        # Calculate how much to read
        to_read = min(len(b), self._total_size - self._position)
        end_pos = self._position + to_read

        # Wait for data to be available
        if not self._wait_for_bytes(self._position, end_pos):
            raise IOError(
                f"Timeout waiting for torrent data at position {self._position}"
            )

        # Read from file
        with open(self._file_path, "rb") as f:
            f.seek(self._position)
            data = f.read(to_read)

        n = len(data)
        b[:n] = data
        self._position += n
        return n

    def read(self, size: int = -1) -> bytes:
        """Read and return bytes."""
        if self._closed:
            raise ValueError("I/O operation on closed file")

        if size < 0:
            size = self._total_size - self._position

        if self._position >= self._total_size:
            return b""

        to_read = min(size, self._total_size - self._position)
        end_pos = self._position + to_read

        # Wait for data
        if not self._wait_for_bytes(self._position, end_pos):
            raise IOError(
                f"Timeout waiting for torrent data at position {self._position}"
            )

        # Read from file
        with open(self._file_path, "rb") as f:
            f.seek(self._position)
            data = f.read(to_read)

        self._position += len(data)
        return data

    def close(self) -> None:
        """Clean up torrent session."""
        if not self._closed:
            self._closed = True
            if self._handle.is_valid():
                self._session.remove_torrent(self._handle)
            if self._temp_dir:
                self._temp_dir.cleanup()

    def __enter__(self) -> "TorrentStream":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def open_torrent_stream(
    url: str,
    save_dir: pathlib.Path | None = None,
    max_download_rate: int = 0,
    prefetch_pieces: int = 32,
) -> TorrentStream:
    """
    Open a torrent URL for streaming.

    Args:
        url: URL to the .torrent file (or data URL with .torrent appended)
        save_dir: Directory to save downloaded data (default: temp dir)
        max_download_rate: Max download speed in bytes/sec (0 = unlimited)
        prefetch_pieces: Number of pieces to prefetch ahead (default 32 = ~32 MB)

    Returns:
        TorrentStream object that can be used as a file-like object

    Example:
        with open_torrent_stream("https://example.com/file.zst.torrent") as stream:
            # stream is a file-like object
            data = stream.read(1024)
    """
    # Ensure URL ends with .torrent
    torrent_url = url if url.endswith(".torrent") else url + ".torrent"
    return TorrentStream(
        torrent_url, save_dir, max_download_rate, prefetch_pieces=prefetch_pieces
    )


# Months with known data issues - skip entirely
# See doc/game-extraction-design.md for full details on each issue
BLOCKLISTED_MONTHS: dict[str, frozenset[str]] = {
    "standard": frozenset(
        [
            "2021-03",  # Datacenter fire, incorrect game results
        ]
    ),
    "chess960": frozenset(
        [
            "2021-03",  # Datacenter fire, incorrect game results
            "2023-11",  # Invalid castling rights in starting FEN for rematches
        ]
    ),
}


def get_blocklist(variant: str) -> frozenset[str]:
    """Get blocklisted months for a variant."""
    return BLOCKLISTED_MONTHS.get(variant, frozenset())


def parse_rate_limit(s: str | None) -> float | None:
    """
    Parse rate limit string to bytes per second.

    Examples: '10M' -> 10_000_000, '500K' -> 500_000, '1G' -> 1_000_000_000
    """
    if not s:
        return None

    s = s.strip().upper()
    multipliers = {
        "K": 1_000,
        "M": 1_000_000,
        "G": 1_000_000_000,
        "KB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
    }

    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) * mult
            except ValueError:
                pass

    # Try parsing as plain number (bytes)
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Invalid rate limit: {s}. Use e.g. '10M', '500K', '1G'")


def get_remote_size(url: str) -> int | None:
    """Get file size from server via HEAD request."""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                return int(content_length)
    except urllib.error.URLError as e:
        print(f"Warning: Could not get file size: {e}")
    return None


def fetch_text(url: str) -> str | None:
    """Fetch text content from URL."""
    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"Error fetching {url}: {e}")
        return None


# Cache directory for games list
CACHE_DIR = pathlib.Path(__file__).parent / ".cache"
GAMES_LIST_CACHE_HOURS = 24  # Refresh cache after 24 hours


def get_games_list(
    variant: str = "standard", use_cache: bool = True
) -> list[tuple[str, str]]:
    """
    Get list of available game archives.

    Returns list of (month, url) tuples, sorted newest first.
    Uses a local cache to avoid repeated network requests.

    Args:
        variant: Game variant ("standard" or "chess960")
        use_cache: If True, use cached list if available and not expired
    """
    cache_file = CACHE_DIR / f"games_list_{variant}.txt"

    # Try to use cache
    if use_cache and cache_file.exists():
        cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if cache_age_hours < GAMES_LIST_CACHE_HOURS:
            try:
                content = cache_file.read_text()
                results = []
                for line in content.strip().split("\n"):
                    url = line.strip()
                    if not url:
                        continue
                    match = re.search(r"(\d{4}-\d{2})\.pgn\.zst$", url)
                    if match:
                        month = match.group(1)
                        results.append((month, url))
                if results:
                    return results
            except Exception:
                pass  # Fall through to fetch from network

    # Fetch from network
    list_url = GAMES_LIST_URL.format(variant=variant)
    content = fetch_text(list_url)
    if not content:
        # If network fails, try to use stale cache
        if cache_file.exists():
            try:
                content = cache_file.read_text()
                print(f"Warning: Using stale cache for {variant} games list")
            except Exception:
                return []
        else:
            return []

    # Parse results
    results = []
    for line in content.strip().split("\n"):
        url = line.strip()
        if not url:
            continue
        # Extract month from URL like lichess_db_standard_rated_2024-01.pgn.zst
        match = re.search(r"(\d{4}-\d{2})\.pgn\.zst$", url)
        if match:
            month = match.group(1)
            results.append((month, url))

    # Save to cache
    if results:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(content)
        except Exception:
            pass  # Cache write failure is not critical

    return results  # Already sorted newest first from Lichess


def get_games_hashes(variant: str = "standard") -> dict[str, str]:
    """
    Get SHA256 hashes for game archives.

    Returns dict mapping filename to hash.
    """
    hash_url = GAMES_HASH_URL.format(variant=variant)
    content = fetch_text(hash_url)
    if not content:
        return {}

    hashes = {}
    for line in content.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            sha256 = parts[0]
            filename = parts[1].lstrip("*")  # Remove leading * if present
            hashes[filename] = sha256

    return hashes


def verify_sha256(path: pathlib.Path, expected_hash: str) -> bool:
    """Verify file SHA256 hash."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    return actual.lower() == expected_hash.lower()


def download_file(
    url: str,
    output_path: pathlib.Path,
    expected_hash: str | None = None,
    chunk_size: int = 1024 * 1024,  # 1 MB chunks
    resume: bool = True,
    rate_limit: float | None = None,  # bytes per second, None = unlimited
    max_retries: int = MAX_RETRIES,
) -> bool:
    """
    Download a file with progress bar, resume support, and retry logic.

    Args:
        url: URL to download from
        output_path: Local path to save the file
        expected_hash: Optional SHA256 hash to verify
        chunk_size: Size of chunks to download
        resume: Whether to resume partial downloads
        rate_limit: Max download speed in bytes/sec (None = unlimited)
        max_retries: Maximum number of retry attempts for transient failures

    Returns:
        True if download completed successfully
    """
    # Get remote file size (with retry)
    remote_size = None
    for attempt in range(max_retries):
        try:
            remote_size = get_remote_size(url)
            break
        except RETRYABLE_EXCEPTIONS as e:
            if attempt < max_retries - 1:
                delay = min(RETRY_DELAY_BASE * (2**attempt), RETRY_DELAY_MAX)
                print(f"Failed to get file size: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Failed to get file size after {max_retries} attempts")

    # Check for existing complete download
    if output_path.exists():
        local_size = output_path.stat().st_size
        if remote_size and local_size >= remote_size:
            # Verify hash if provided
            if expected_hash:
                print(f"Verifying hash for {output_path.name}...")
                if verify_sha256(output_path, expected_hash):
                    print(f"File already complete and verified: {output_path}")
                    return True
                else:
                    print(f"Hash mismatch! Re-downloading {output_path.name}")
            else:
                print(f"File already complete: {output_path}")
                return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Retry loop for download
    for attempt in range(max_retries):
        try:
            # Check for existing partial download (re-check each attempt)
            start_byte = 0
            mode = "wb"

            if resume and output_path.exists():
                local_size = output_path.stat().st_size
                if remote_size and 0 < local_size < remote_size:
                    if attempt == 0:
                        print(f"Resuming download from byte {local_size:,}")
                    start_byte = local_size
                    mode = "ab"

            # Prepare request with Range header for resume
            request = urllib.request.Request(url)
            if start_byte > 0:
                request.add_header("Range", f"bytes={start_byte}-")

            with urllib.request.urlopen(request, timeout=60) as response:
                # Check if server supports range requests
                if start_byte > 0:
                    if response.status != 206:  # Partial Content
                        print("Server doesn't support resume, starting from beginning")
                        start_byte = 0
                        mode = "wb"

                # Get total size for progress bar
                content_length = response.headers.get("Content-Length")
                if content_length:
                    total_size = int(content_length) + start_byte
                elif remote_size:
                    total_size = remote_size
                else:
                    total_size = None

                # Download with progress bar
                with open(output_path, mode) as f:
                    with tqdm.tqdm(
                        total=total_size,
                        initial=start_byte,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=output_path.name,
                    ) as pbar:
                        while True:
                            chunk_start = time.monotonic()
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))

                            # Rate limiting
                            if rate_limit:
                                chunk_time = time.monotonic() - chunk_start
                                expected_time = len(chunk) / rate_limit
                                if chunk_time < expected_time:
                                    time.sleep(expected_time - chunk_time)

            # Verify final size
            if remote_size:
                final_size = output_path.stat().st_size
                if final_size != remote_size:
                    print(
                        f"Warning: Size mismatch! Expected {remote_size:,}, got {final_size:,}"
                    )
                    # Don't retry on size mismatch - likely a server issue
                    return False

            # Verify hash if provided
            if expected_hash:
                print("Verifying hash...")
                if not verify_sha256(output_path, expected_hash):
                    print("Hash verification failed!")
                    return False
                print("Hash verified.")

            return True

        except RETRYABLE_EXCEPTIONS as e:
            if attempt < max_retries - 1:
                delay = min(RETRY_DELAY_BASE * (2**attempt), RETRY_DELAY_MAX)
                print(f"\nDownload error: {e}")
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed. Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                print(f"\nDownload failed after {max_retries} attempts: {e}")
                return False

        except KeyboardInterrupt:
            print("\nDownload interrupted. Run again to resume.")
            return False

    return False  # Should not reach here


def download_torrent(
    torrent_url: str,
    output_path: pathlib.Path,
    expected_hash: str | None = None,
    timeout: int = 3600,  # 1 hour default
    max_download_rate: int = 0,  # 0 = unlimited (bytes/sec)
    max_upload_rate: int = 100 * 1024,  # 100 KB/s upload by default
) -> bool:
    """
    Download a file via BitTorrent.

    Args:
        torrent_url: URL to .torrent file
        output_path: Local path to save the downloaded file
        expected_hash: Optional SHA256 hash to verify
        timeout: Max seconds to wait for download (default 1 hour)
        max_download_rate: Max download speed in bytes/sec (0 = unlimited)
        max_upload_rate: Max upload speed in bytes/sec (default 100 KB/s)

    Returns:
        True if download completed successfully
    """
    # Check if file already exists and is complete
    if output_path.exists() and expected_hash:
        print(f"Verifying existing file: {output_path.name}...")
        if verify_sha256(output_path, expected_hash):
            print(f"File already complete and verified: {output_path}")
            return True
        else:
            print(f"Hash mismatch, re-downloading...")
            output_path.unlink()

    # Fetch torrent file
    print(f"Fetching torrent: {torrent_url}")
    try:
        with urllib.request.urlopen(torrent_url, timeout=30) as resp:
            torrent_data = resp.read()
    except urllib.error.URLError as e:
        print(f"Failed to fetch torrent: {e}")
        return False

    # Parse torrent
    try:
        info = lt.torrent_info(lt.bdecode(torrent_data))
    except Exception as e:
        print(f"Failed to parse torrent: {e}")
        return False

    total_size = info.total_size()
    print(f"File: {info.name()} ({total_size / 1e9:.2f} GB)")

    # Create session with settings
    ses = lt.session()
    settings = {
        "download_rate_limit": max_download_rate,
        "upload_rate_limit": max_upload_rate,
        "connections_limit": 200,
        "active_downloads": 1,
        "active_seeds": 1,
    }
    ses.apply_settings(settings)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dir = output_path.parent

    # Check for partial download (libtorrent resume)
    resume_file = output_path.with_suffix(output_path.suffix + ".resume")
    resume_data = None
    if resume_file.exists():
        try:
            with open(resume_file, "rb") as f:
                resume_data = lt.bdecode(f.read())
            print(f"Resuming from checkpoint...")
        except Exception:
            pass

    # Add torrent
    params = lt.add_torrent_params()
    params.ti = info
    params.save_path = str(save_dir)
    params.flags |= lt.torrent_flags.sequential_download  # Download in order
    if resume_data:
        params.resume_data = lt.bencode(resume_data)

    h = ses.add_torrent(params)

    print(f"Starting download (timeout: {timeout}s)...")
    start_time = time.monotonic()
    last_progress = -1

    try:
        with tqdm.tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=output_path.name,
        ) as pbar:
            while True:
                s = h.status()

                # Update progress bar
                downloaded = int(s.progress * total_size)
                pbar.n = downloaded
                pbar.set_postfix(
                    {
                        "down": f"{s.download_rate / 1024:.0f}KB/s",
                        "peers": s.num_peers,
                        "seeds": s.num_seeds,
                    }
                )
                pbar.refresh()

                # Check if complete
                if s.is_seeding or s.progress >= 1.0:
                    pbar.n = total_size
                    pbar.refresh()
                    break

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    print(f"\nDownload timed out after {timeout}s")
                    # Save resume data for next attempt
                    h.save_resume_data()
                    time.sleep(1)
                    alerts = ses.pop_alerts()
                    for alert in alerts:
                        if isinstance(alert, lt.save_resume_data_alert):
                            with open(resume_file, "wb") as f:
                                f.write(lt.bencode(alert.resume_data))
                            print(f"Resume data saved to {resume_file}")
                    return False

                # Check for errors
                if s.errc.value() != 0:
                    print(f"\nTorrent error: {s.errc.message()}")
                    return False

                time.sleep(1)

        # Download complete
        print(f"\nDownload complete!")

        # Clean up resume file
        if resume_file.exists():
            resume_file.unlink()

        # Verify hash
        if expected_hash:
            print("Verifying hash...")
            actual_path = save_dir / info.name()
            if actual_path != output_path:
                actual_path.rename(output_path)
            if not verify_sha256(output_path, expected_hash):
                print("Hash verification failed!")
                return False
            print("Hash verified.")

        return True

    except KeyboardInterrupt:
        print("\nDownload interrupted. Saving resume data...")
        h.save_resume_data()
        time.sleep(1)
        alerts = ses.pop_alerts()
        for alert in alerts:
            if isinstance(alert, lt.save_resume_data_alert):
                with open(resume_file, "wb") as f:
                    f.write(lt.bencode(alert.resume_data))
                print(f"Resume data saved. Run again to continue.")
        return False

    finally:
        ses.remove_torrent(h)


def download_dataset(name: str, output_dir: pathlib.Path, force: bool = False) -> bool:
    """Download a specific dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False

    dataset = DATASETS[name]
    output_path = output_dir / dataset["filename"]

    print(f"\nDownloading {name}: {dataset['description']}")
    print(f"URL: {dataset['url']}")
    print(f"Output: {output_path}")

    # Check if file exists and is complete
    if output_path.exists() and not force:
        remote_size = get_remote_size(dataset["url"])
        local_size = output_path.stat().st_size
        if remote_size and local_size >= remote_size:
            print(f"Already downloaded: {output_path}")
            return True

    return download_file(dataset["url"], output_path)


def download_games_month(
    month: str,
    output_dir: pathlib.Path,
    variant: str = "standard",
    hashes: dict[str, str] | None = None,
    force: bool = False,
    rate_limit: float | None = None,
    use_torrent: bool = False,
) -> pathlib.Path | None:
    """
    Download a single month's game archive.

    Args:
        month: Month in YYYY-MM format
        output_dir: Directory to save the file
        variant: "standard" or "chess960"
        hashes: Dict mapping filename to SHA256 hash
        force: Force re-download even if file exists
        rate_limit: Max download speed in bytes/sec (None = unlimited)
        use_torrent: Use BitTorrent instead of HTTP (recommended for large files)

    Returns:
        Path to downloaded file, or None if failed/skipped.
    """
    blocklist = get_blocklist(variant)
    if month in blocklist:
        print(f"Skipping blocklisted month: {month} ({variant})")
        return None

    # Get URL from list
    games_list = get_games_list(variant)
    url = None
    for m, u in games_list:
        if m == month:
            url = u
            break

    if not url:
        print(f"Month {month} not found in {variant} archives")
        return None

    filename = url.split("/")[-1]
    output_path = output_dir / filename

    # Get hash if available
    expected_hash = None
    if hashes:
        expected_hash = hashes.get(filename)

    if output_path.exists() and not force:
        # Check if complete
        remote_size = get_remote_size(url)
        local_size = output_path.stat().st_size
        if remote_size and local_size >= remote_size:
            if expected_hash:
                print(f"Verifying {filename}...")
                if verify_sha256(output_path, expected_hash):
                    print(f"Already downloaded and verified: {output_path}")
                    return output_path
            else:
                print(f"Already downloaded: {output_path}")
                return output_path

    print(f"\nDownloading {month} ({variant})...")

    if use_torrent:
        torrent_url = url + ".torrent"
        print(f"Torrent: {torrent_url}")
        max_rate = int(rate_limit) if rate_limit else 0
        if download_torrent(
            torrent_url, output_path, expected_hash, max_download_rate=max_rate
        ):
            return output_path
        print("Torrent download failed, falling back to HTTP...")

    print(f"URL: {url}")
    if rate_limit:
        print(f"Rate limit: {rate_limit / 1e6:.1f} MB/s")

    if download_file(url, output_path, expected_hash, rate_limit=rate_limit):
        return output_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Lichess training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Static datasets
    uv run python download.py puzzles              # Download puzzle database
    uv run python download.py evals                # Download evaluation database

    # Game archives
    uv run python download.py games --list         # List available months
    uv run python download.py games --month 2024-01    # Download one month
    uv run python download.py games --all          # Download ALL months (huge!)
    uv run python download.py games --after 2023-01    # Download 2023-01 onwards
    uv run python download.py games --before 2020-01   # Download before 2020-01

    # Download and process
    uv run python download.py games --month 2024-01 --process -e /path/to/stockfish

Datasets:
    puzzles  - Lichess puzzle database (~287 MB)
    evals    - Lichess evaluation database (~2.5 GB)
    games    - Monthly game archives (50-100+ GB each, ~10+ TB total)
        """,
    )
    parser.add_argument(
        "dataset",
        choices=["puzzles", "evals", "games", "all"],
        help="Dataset to download",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (default: . for static, data/games for archives)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume partial downloads, start fresh",
    )

    # Games-specific options
    games_group = parser.add_argument_group("games options")
    games_group.add_argument(
        "--list",
        action="store_true",
        help="List available months",
    )
    games_group.add_argument(
        "--month",
        type=str,
        help="Download specific month (YYYY-MM)",
    )
    games_group.add_argument(
        "--all",
        action="store_true",
        dest="all_months",
        help="Download all available months",
    )
    games_group.add_argument(
        "--after",
        type=str,
        metavar="YYYY-MM",
        help="Download months >= this date",
    )
    games_group.add_argument(
        "--before",
        type=str,
        metavar="YYYY-MM",
        help="Download months < this date",
    )
    games_group.add_argument(
        "--variant",
        choices=["standard", "chess960"],
        default="standard",
        help="Game variant (default: standard)",
    )
    games_group.add_argument(
        "--verify",
        action="store_true",
        help="Verify SHA256 hashes",
    )
    games_group.add_argument(
        "--rate-limit",
        type=str,
        default=None,
        metavar="SPEED",
        help="Limit download speed (e.g., '10M' for 10 MB/s, '500K' for 500 KB/s)",
    )
    games_group.add_argument(
        "--torrent",
        action="store_true",
        help="Use BitTorrent for download (recommended for large files, more reliable)",
    )

    # Processing options
    process_group = parser.add_argument_group("processing options")
    process_group.add_argument(
        "--process",
        action="store_true",
        help="Process downloaded files with preprocess.py",
    )
    process_group.add_argument(
        "-e",
        "--engine",
        type=str,
        help="Path to UCI engine (required for --process)",
    )
    process_group.add_argument(
        "-n",
        "--max-positions",
        type=int,
        default=None,
        help="Max positions to extract per file",
    )
    process_group.add_argument(
        "-d",
        "--depth",
        type=int,
        default=12,
        help="Engine search depth (default: 12)",
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        if args.dataset == "games":
            args.output_dir = pathlib.Path("data/games")
        else:
            args.output_dir = pathlib.Path(".")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Handle games dataset
    if args.dataset == "games":
        games_list = get_games_list(args.variant)
        if not games_list:
            print("Failed to fetch games list")
            sys.exit(1)

        # List mode
        if args.list:
            print(
                f"\nAvailable {args.variant} game archives ({len(games_list)} months):\n"
            )
            total_size = 0
            for month, url in games_list:
                size = get_remote_size(url)
                size_str = f"{size / 1e9:.1f} GB" if size else "? GB"
                if size:
                    total_size += size
                blocklist = get_blocklist(args.variant)
                blocklisted = " [BLOCKLISTED]" if month in blocklist else ""
                print(f"  {month}: {size_str}{blocklisted}")
            print(f"\nTotal: ~{total_size / 1e12:.1f} TB")
            blocklist = get_blocklist(args.variant)
            print(
                f"\nBlocklisted months (data issues): {', '.join(sorted(blocklist)) or 'none'}"
            )
            return

        # Determine which months to download
        months_to_download = []

        if args.month:
            months_to_download = [args.month]
        elif args.all_months:
            months_to_download = [m for m, _ in games_list]
        elif args.after or args.before:
            for month, _ in games_list:
                if args.after and month < args.after:
                    continue
                if args.before and month >= args.before:
                    continue
                months_to_download.append(month)
        else:
            print("Error: Specify --month, --all, --after, or --before")
            print("Use --list to see available months")
            sys.exit(1)

        # Filter out blocklisted months
        blocklist = get_blocklist(args.variant)
        original_count = len(months_to_download)
        months_to_download = [m for m in months_to_download if m not in blocklist]
        if len(months_to_download) < original_count:
            skipped = original_count - len(months_to_download)
            print(f"Skipping {skipped} blocklisted month(s)")

        if not months_to_download:
            print("No months to download")
            sys.exit(0)

        print(
            f"\nWill download {len(months_to_download)} month(s): {', '.join(months_to_download[:5])}",
            end="",
        )
        if len(months_to_download) > 5:
            print(f" ... and {len(months_to_download) - 5} more")
        else:
            print()

        # Get hashes if verifying
        hashes = {}
        if args.verify:
            print("Fetching SHA256 hashes...")
            hashes = get_games_hashes(args.variant)
            print(f"Got hashes for {len(hashes)} files")

        # Parse rate limit
        rate_limit = parse_rate_limit(args.rate_limit)
        if rate_limit:
            print(f"Rate limit: {rate_limit / 1e6:.1f} MB/s")
        if args.torrent:
            print("Using BitTorrent (recommended for large files)")

        # Download each month
        downloaded = []
        failed = []

        for month in months_to_download:
            result = download_games_month(
                month,
                args.output_dir,
                args.variant,
                hashes,
                args.force,
                rate_limit,
                use_torrent=args.torrent,
            )
            if result:
                downloaded.append((month, result))
            else:
                failed.append(month)

        print(f"\n{'=' * 60}")
        print(f"Downloaded: {len(downloaded)}, Failed: {len(failed)}")
        if failed:
            print(f"Failed months: {', '.join(failed)}")

        # Process if requested
        if args.process and downloaded:
            if not args.engine:
                print("Error: --engine is required for --process")
                sys.exit(1)

            print(f"\nProcessing {len(downloaded)} file(s)...")
            for month, path in downloaded:
                output = args.output_dir / f"games_{month.replace('-', '_')}.parquet"
                cmd = [
                    "uv",
                    "run",
                    "python",
                    "preprocess.py",
                    "games",
                    "-i",
                    str(path),
                    "-e",
                    args.engine,
                    "-d",
                    str(args.depth),
                    "-o",
                    str(output),
                ]
                if args.max_positions:
                    cmd.extend(["-n", str(args.max_positions)])

                print(f"\nProcessing {month}...")
                print(f"  Command: {' '.join(cmd)}")
                subprocess.run(cmd)

        return

    # Handle static datasets (puzzles, evals, all)
    datasets_to_download = (
        list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    )

    success = True
    for name in datasets_to_download:
        if not download_dataset(name, args.output_dir, args.force):
            success = False

    if success:
        print("\nDownload complete!")
        print("\nNext steps:")
        print("  # Process puzzles (requires UCI engine for evaluation):")
        print(
            "  uv run python preprocess.py puzzles -i lichess_db_puzzle.csv.zst -e /path/to/stockfish -o data/puzzles.parquet"
        )
        print("\n  # Process evaluations:")
        print(
            "  uv run python preprocess.py evals -i lichess_db_eval.jsonl.zst -o data/evals.parquet"
        )
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
