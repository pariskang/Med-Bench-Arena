"""Asset fetching: download (with resume-ish skip) and extract archives.

For the multimodal sets whose images ship as a separate ``images.zip`` (OmniMedVQA,
PMC-VQA, MedBookVQA, MedXpertQA-MM, TCM-Vision-Benchmark, SLAKE-bilingual): point
the dataset's ``image_zip`` at the archive URL and ``image_base`` at where it should
live; the adapter calls :func:`ensure_image_base` once to download + unzip, then
prepends the extracted dir to each relative image path. Idempotent via a marker file.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

CACHE_DIR = Path(os.environ.get("MEDEVAL_CACHE", "data/cache"))
_MARKER = ".medeval_extracted"


def _download_stream(url: str, dest: Path, retries: int = 6, chunk: int = 1 << 20) -> Path:
    """Robustly stream a (possibly huge) file to ``dest``.

    Resilient to proxies/CDNs that cap or truncate a single response: it RESUMES
    from the partial ``.part`` via HTTP ``Range`` requests, so each round makes
    forward progress. Only *no-progress* rounds count against ``retries``; the file
    is renamed into place atomically and only when complete, so a failed download
    never poisons the cache. ``dest`` existing ⇒ a prior verified-complete download.
    """
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = 0
    stalls = 0
    last = -1
    rounds = 0
    while True:
        rounds += 1
        done = tmp.stat().st_size if tmp.exists() else 0
        stalls = stalls + 1 if done == last else 0   # progress resets the budget
        last = done
        if stalls > retries or rounds > 500:
            raise OSError(f"download stalled at {done}/{total or '?'} bytes: {url}")
        try:
            headers = {"User-Agent": "medeval/1.0"}
            if done:
                headers["Range"] = f"bytes={done}-"
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=120) as r:
                status = getattr(r, "status", r.getcode())
                crange = r.headers.get("Content-Range")
                if crange and "/" in crange:                 # "bytes a-b/total"
                    total = int(crange.rsplit("/", 1)[-1])
                elif status != 206:
                    total = int(r.headers.get("Content-Length") or 0)
                resuming = bool(done) and status == 206
                if not resuming:                             # server ignored Range → restart
                    done = 0
                next_mark = (done // (50 << 20) + 1) * (50 << 20)
                with open(tmp, "ab" if resuming else "wb") as f:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if done >= next_mark:
                            pct = f" ({100*done//total}%)" if total else ""
                            print(f"[medeval] downloading {dest.name}: {done >> 20} MiB{pct}")
                            next_mark += 50 << 20
            if not total or done >= total:                   # complete
                tmp.replace(dest)
                return dest
        except Exception as e:  # noqa: BLE001 — resume on the next round
            got = tmp.stat().st_size if tmp.exists() else 0
            print(f"[medeval] resuming {dest.name} from {got} bytes ({e})")
            time.sleep(min(2 ** stalls, 8))


def _obtain(src: str | Path) -> Path:
    """Return a local path to ``src`` (download if it is a URL)."""
    s = str(src)
    if s.startswith(("http://", "https://")):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        name = s.split("/")[-1].split("?")[0] or "asset"
        return _download_stream(s, CACHE_DIR / name)
    return Path(s)


def _extract(archive: Path, dest: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest)  # noqa: S202 (trusted dataset archives)
    else:  # not an archive — just place the file inside dest
        shutil.copy2(archive, dest / archive.name)


def ensure_extracted(src: str | Path, dest: str | Path,
                     marker_name: str = _MARKER) -> Path:
    """Download (if URL) + extract ``src`` into ``dest`` once. Returns ``dest``.

    A marker file under ``dest`` makes this a no-op on subsequent calls.
    """
    dest = Path(dest)
    marker = dest / marker_name
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[medeval] preparing assets in {dest} from {src}")
    _extract(_obtain(src), dest)
    marker.write_text(str(src), encoding="utf-8")
    return dest


def ensure_image_base(zip_src: str | Path, image_base: str | Path | None) -> str:
    """Extract ``zip_src`` and return the directory to use as ``image_base``
    (with a trailing separator so relative image paths concatenate cleanly)."""
    dest = Path(image_base) if image_base else (CACHE_DIR / "images" /
                                                Path(str(zip_src)).stem)
    ensure_extracted(zip_src, dest)
    return str(dest).rstrip("/") + "/"
