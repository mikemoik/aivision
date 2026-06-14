"""Resilient downloader for the RLMD dataset Google-Drive folder.

`gdown --folder` aborts the whole pass on the FIRST file Google rate-limits
("Cannot retrieve the public link ... had many accesses"), so it never gets
past the few-dozen-file public-download quota window. This script instead:

  1. enumerates the folder once (folder API, no per-file quota cost),
  2. downloads each file individually, SKIPPING ones already on disk,
  3. CONTINUES past a rate-limited file instead of aborting,
  4. backs off when it hits a run of consecutive failures (quota window), and
  5. loops in rounds, re-trying only the still-missing files, until complete.

Source folder: https://github.com/stu9113611/RLMD  (RLMD, ICCE-TW 2023).
Re-runnable + idempotent: re-running only fetches what is still missing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/18-Lnz2MaPQWSOJ8aK1ikkwXLLdjImYOA"


def enumerate_files(url: str, out: Path):
    """Folder listing -> [(file_id, local_path)] without downloading anything."""
    files = gdown.download_folder(url=url, output=str(out), skip_download=True,
                                  quiet=True)
    return [(f.id, Path(f.local_path)) for f in files]


def missing(files):
    return [(fid, p) for fid, p in files if not (p.exists() and p.stat().st_size > 0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=FOLDER_URL)
    ap.add_argument("--out", type=Path, default=Path("data/rlmd"))
    ap.add_argument("--rounds", type=int, default=200,
                    help="max retry rounds over the still-missing set")
    ap.add_argument("--cooldown", type=int, default=120,
                    help="seconds to sleep after a run of consecutive failures")
    ap.add_argument("--fail-streak", type=int, default=8,
                    help="consecutive failures that trigger a cooldown (quota hit)")
    ap.add_argument("--hard-ban", type=int, default=24,
                    help="consecutive failures with 0 success this round -> assume a "
                         "hard quota ban, abort the round and long-sleep")
    ap.add_argument("--ban-sleep", type=int, default=1800,
                    help="seconds to sleep when a hard quota ban is detected")
    args = ap.parse_args()

    print(f"[rlmd] enumerating {args.url} ...", flush=True)
    files = enumerate_files(args.url, args.out)
    print(f"[rlmd] {len(files)} files in folder", flush=True)

    for rnd in range(1, args.rounds + 1):
        todo = missing(files)
        print(f"[rlmd] round {rnd}: {len(files) - len(todo)}/{len(files)} present, "
              f"{len(todo)} missing", flush=True)
        if not todo:
            print("[rlmd] COMPLETE — all files present", flush=True)
            return
        got = streak = 0
        banned = False
        for fid, p in todo:
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                r = gdown.download(id=fid, output=str(p), quiet=True)
            except Exception:
                r = None
            if r and p.exists() and p.stat().st_size > 0:
                got += 1
                streak = 0
            else:
                streak += 1
                # nothing downloaded yet this round and a long fail run -> hard ban:
                # bail out instead of grinding through thousands of doomed requests
                if got == 0 and streak >= args.hard_ban:
                    banned = True
                    break
                if streak >= args.fail_streak:
                    print(f"[rlmd]   quota hit ({streak} fails in a row) — "
                          f"got {got} this round; cooling down {args.cooldown}s",
                          flush=True)
                    time.sleep(args.cooldown)
                    streak = 0
        print(f"[rlmd] round {rnd} done: +{got} files"
              f"{' (HARD BAN detected)' if banned else ''}", flush=True)
        if got == 0:
            print(f"[rlmd]   quota ban — sleeping {args.ban_sleep}s before next round",
                  flush=True)
            time.sleep(args.ban_sleep)

    todo = missing(files)
    print(f"[rlmd] STOP after {args.rounds} rounds: {len(todo)} still missing",
          flush=True)


if __name__ == "__main__":
    main()
