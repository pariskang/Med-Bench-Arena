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


def _download_stream(url: str, dest: Path, retries: int = 4, chunk: int = 1 << 20) -> Path:
    """Stream a (possibly huge) file to ``dest``, skipping if already complete."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "medeval/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                total = int(r.headers.get("Content-Length") or 0)
                if dest.exists() and total and dest.stat().st_size == total:
                    return dest
                done = 0
                next_mark = 50 << 20
                with open(tmp, "wb") as f:
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
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                raise
            print(f"[medeval] download retry {attempt + 1} ({e})")
            time.sleep(2 ** attempt)
    return dest


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
