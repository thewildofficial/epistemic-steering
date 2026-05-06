"""Download generation-time probe data from Modal volume.

Usage:
    python scripts/download_gen_time_data.py

Downloads all ``gen_time_batch_*.pkl`` and ``gen_time_all.pkl`` files from
``epistemic-model-cache/results/gen_time/`` to ``data/gen_time/``.
"""

from pathlib import Path

import modal

VOLUME_NAME = "epistemic-model-cache"
REMOTE_PREFIX = "results/gen_time"
LOCAL_DIR = Path("data/gen_time")


def download(force: bool = False) -> int:
    """Download all generation-time data from Modal volume.

    Args:
        force: Re-download even if files already exist locally.

    Returns:
        Number of files downloaded.
    """
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Modal volume: {VOLUME_NAME} ...")
    vol = modal.Volume.from_name(VOLUME_NAME)

    print(f"Listing {REMOTE_PREFIX}/ ...")
    try:
        entries = list(vol.listdir(REMOTE_PREFIX))
    except Exception as exc:
        print(f"Error listing remote directory: {exc}")
        return 0

    print(f"Found {len(entries)} entries")

    downloaded = 0
    skipped = 0

    for entry in entries:
        filename = entry.path.split("/")[-1]
        if not filename:
            continue

        local_path = LOCAL_DIR / filename

        if local_path.exists() and not force:
            skipped += 1
            continue

        try:
            chunks = list(vol.read_file(entry.path))
            data = b"".join(chunks)
            with open(local_path, "wb") as f:
                f.write(data)
            downloaded += 1
            print(f"  Downloaded {filename}")
        except Exception as exc:
            print(f"  ERROR downloading {filename}: {exc}")

    print(f"\nDownloaded {downloaded} files, skipped {skipped} existing")
    print(f"Local directory: {LOCAL_DIR.resolve()}")
    return downloaded


if __name__ == "__main__":
    download()
