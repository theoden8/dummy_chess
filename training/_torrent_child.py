"""
Standalone torrent-streaming child process. Imports ONLY libtorrent + stdlib so
it isn't poisoned by pyarrow (which breaks libtorrent's HTTP web-seed in the
parent process).

Usage:
    python _torrent_child.py <url> [save_dir]

If save_dir is omitted the child uses a private TemporaryDirectory and cleans
it up on exit. With save_dir, the file is left behind for caching across runs.

Reads bytes from the streaming torrent, pumps them to stdout as they arrive.
Parent reads stdout to get a file-like stream, pipes through zstd reader, etc.

Logs (status, errors) go to stderr.
"""
import pathlib
import sys
import tempfile
import time
import urllib.request

import libtorrent as lt


CHUNK = 1 << 16  # 64 KB pipe write
PIECE_TIMEOUT = 300


def main(url: str, save_dir: str | None = None) -> int:
    if not url.endswith(".torrent"):
        url = url + ".torrent"
    sys.stderr.write(f"[child] fetching {url}\n")
    sys.stderr.flush()
    with urllib.request.urlopen(url, timeout=30) as r:
        tdata = r.read()
    info = lt.torrent_info(lt.bdecode(tdata))
    total = info.total_size()
    piece_len = info.piece_length()
    num_pieces = info.num_pieces()
    sys.stderr.write(
        f"[child] {info.name()}  size={total/1e9:.2f} GB  pieces={num_pieces}\n"
    )
    sys.stderr.flush()

    ses = lt.session()
    ses.apply_settings({
        "connections_limit": 20,
        "active_downloads": 1,
        "active_seeds": 1,
        "cache_size": 16,
        "max_queued_disk_bytes": 128 * 1024,
        "disk_write_mode": int(lt.mmap_write_mode_t.always_pwrite),
        "send_buffer_watermark": 64 * 1024,
        "max_peer_recv_buffer_size": 256 * 1024,
        "enable_dht": False,
        "enable_lsd": False,
        "enable_upnp": False,
        "enable_natpmp": False,
    })

    td: tempfile.TemporaryDirectory | None
    if save_dir is None:
        td = tempfile.TemporaryDirectory()
        save_path = td.name
    else:
        td = None
        save_path = save_dir
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    params = lt.add_torrent_params()
    params.ti = info
    params.save_path = save_path
    params.flags |= lt.torrent_flags.sequential_download
    handle = ses.add_torrent(params)
    file_path = pathlib.Path(save_path) / info.name()

    while not handle.status().has_metadata:
        time.sleep(0.05)

    pos = 0
    last_log = time.monotonic()
    while pos < total:
        end = min(pos + CHUNK, total)
        # Pieces covering [pos, end)
        first_piece = pos // piece_len
        last_piece = (end - 1) // piece_len
        # Bump priority on the immediately-needed pieces, plus a small prefetch
        for pi in range(first_piece, min(last_piece + 32, num_pieces)):
            if not handle.have_piece(pi):
                handle.piece_priority(pi, 7 if pi <= last_piece else 6)
        # Wait until we have all needed pieces
        deadline = time.monotonic() + PIECE_TIMEOUT
        while True:
            need = [pi for pi in range(first_piece, last_piece + 1) if not handle.have_piece(pi)]
            if not need:
                break
            if time.monotonic() > deadline:
                sys.stderr.write(
                    f"[child] timeout waiting for pieces {need} at pos {pos}\n"
                )
                sys.stderr.flush()
                return 2
            time.sleep(0.05)
        # Read what we waited for and pump to stdout
        with open(file_path, "rb") as f:
            f.seek(pos)
            chunk = f.read(end - pos)
        try:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            sys.stderr.write("[child] parent closed pipe, exiting\n")
            sys.stderr.flush()
            return 0
        pos += len(chunk)
        now = time.monotonic()
        if now - last_log > 30.0:
            st = handle.status()
            sys.stderr.write(
                f"[child] pumped {pos/1e9:.2f}/{total/1e9:.2f} GB  "
                f"peers={st.num_peers}  dl={st.download_rate/1024:.1f}KB/s\n"
            )
            sys.stderr.flush()
            last_log = now

    sys.stderr.write(f"[child] done, pumped {pos} bytes\n")
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.stderr.write(f"usage: {sys.argv[0]} <url> [save_dir]\n")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None))
