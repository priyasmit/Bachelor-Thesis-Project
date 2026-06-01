import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Download a 1000-APK dataset from AndroZoo CSV.")
    p.add_argument("--csv", required=True, help="Path to AndroZoo CSV (must contain sha256, vt_detection, apk_size optional).")
    p.add_argument("--apikey", required=True, help="Your AndroZoo API key.")
    p.add_argument("--out", default="apks", help="Output directory for APKs.")
    p.add_argument("--n", type=int, default=1000, help="Total number of APKs to download.") 
    p.add_argument("--balanced", action="store_true", help="Try to balance benign vs malware 50/50 using vt_detection.")
    p.add_argument("--malware-threshold", type=int, default=1, help="vt_detection >= threshold => malware (default 1).")
    p.add_argument("--max-apk-mb", type=float, default=50.0, help="Max APK size in MB (if apk_size exists).")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between downloads (rate limiting).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    p.add_argument("--retries", type=int, default=3, help="Retries per APK.")
    p.add_argument("--timeout", type=int, default=300, help="Curl timeout seconds per request.")
    return p.parse_args()

def read_csv_rows(csv_path: Path):
    # Uses csv module to handle huge files without loading everything into RAM at once
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for row in reader:
            yield row, header

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def select_hashes(csv_path: Path, n: int, balanced: bool, malware_threshold: int, max_apk_mb: float, seed: int):
    random.seed(seed)

    benign = []
    malware = []
    unknown = []  # in case vt_detection missing

    # First pass: collect candidates
    has_apk_size = False
    has_vt = False

    for row, header in read_csv_rows(csv_path):
        if "sha256" not in row or not row["sha256"]:
            continue
        sha = row["sha256"].strip()

        if "apk_size" in row:
            has_apk_size = True
            apk_size = safe_int(row.get("apk_size", ""), default=None)
            if apk_size is not None:
                if apk_size > int(max_apk_mb * 1024 * 1024):
                    continue

        vt = row.get("vt_detection", None)
        if vt is not None and vt != "":
            has_vt = True
            vt_i = safe_int(vt, default=None)
            if vt_i is None:
                unknown.append(sha)
            elif vt_i >= malware_threshold:
                malware.append(sha)
            else:
                benign.append(sha)
        else:
            unknown.append(sha)

    if balanced and not has_vt:
        raise RuntimeError("balanced sampling requested, but CSV has no vt_detection column.")

    if balanced:
        half = n // 2
        if len(benign) < half or len(malware) < half:
            raise RuntimeError(
                f"Not enough samples for balanced split: benign={len(benign)}, malware={len(malware)}, need {half} each."
            )
        selected = random.sample(benign, half) + random.sample(malware, half)
        if n % 2 == 1:
            # add one extra (prefer benign)
            selected.append(random.choice(benign))
        random.shuffle(selected)
        return selected, {"mode": "balanced", "benign_pool": len(benign), "malware_pool": len(malware)}
    else:
        pool = benign + malware if has_vt else unknown
        if len(pool) < n:
            raise RuntimeError(f"Not enough candidates in pool: {len(pool)} < {n}")
        selected = random.sample(pool, n)
        return selected, {"mode": "random", "pool": len(pool), "has_vt": has_vt, "has_apk_size": has_apk_size}

def already_downloaded(out_dir: Path):
    # Accept .apk files in out_dir; curl uses remote-header-name, filenames can vary.
    # Adittionally keep a sha256->filename map in downloads.csv..
    done_sha = set()
    manifest = out_dir / "downloads.csv"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("sha256") and row.get("status") == "ok":
                    done_sha.add(row["sha256"])
    return done_sha

def append_manifest(out_dir: Path, row: dict):
    manifest = out_dir / "downloads.csv"
    exists = manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sha256", "status", "http_hint", "filename", "attempts", "timestamp"
        ])
        if not exists:
            w.writeheader()
        w.writerow(row)

def download_one(apikey: str, sha256: str, out_dir: Path, retries: int, timeout: int):
    url = "https://androzoo.uni.lu/api/download"

    cmd = [
        "curl", "-fLSs", "--connect-timeout", "30", "--max-time", str(timeout),
        "-O", "--remote-header-name", "-G",
        "-d", f"apikey={apikey}",
        "-d", f"sha256={sha256}",
        url
    ]

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            before = set(p.name for p in out_dir.glob("*.apk"))
            proc = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True)

            if proc.returncode == 0:
                after = set(p.name for p in out_dir.glob("*.apk"))
                new_files = list(after - before)
                filename = new_files[0] if new_files else ""
                return True, "ok", filename, attempt

            # curl failed: keep stderr for hint
            last_err = (proc.stderr or proc.stdout or "").strip()[:200]
        except Exception as e:
            last_err = str(e)[:200]

        time.sleep(min(10, 1.5 * attempt))

    return False, last_err or "curl_failed", "", retries

def write_selection(out_dir: Path, selected):
    sel_path = out_dir / "selected_sha256.txt"
    sel_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select hashes
    selected, info = select_hashes(
        csv_path=Path(args.csv),
        n=args.n,
        balanced=args.balanced,
        malware_threshold=args.malware_threshold,
        max_apk_mb=args.max_apk_mb,
        seed=args.seed
    )
    write_selection(out_dir, selected)

    # Resume
    done_sha = already_downloaded(out_dir)

    print(f"[+] Selection: {len(selected)} APKs | {info}")
    print(f"[+] Output dir: {out_dir.resolve()}")
    print(f"[+] Already downloaded (from manifest): {len(done_sha)}")

    remaining = [sha for sha in selected if sha not in done_sha]
    print(f"[+] Remaining: {len(remaining)}")

    ok = 0
    fail = 0

    for i, sha in enumerate(remaining, start=1):
        success, hint, filename, attempts = download_one(
            apikey=args.apikey,
            sha256=sha,
            out_dir=out_dir,
            retries=args.retries,
            timeout=args.timeout
        )

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if success:
            ok += 1
            append_manifest(out_dir, {
                "sha256": sha,
                "status": "ok",
                "http_hint": "",
                "filename": filename,
                "attempts": attempts,
                "timestamp": ts
            })
            print(f"[OK] {i}/{len(remaining)} {sha} -> {filename}")
        else:
            fail += 1
            append_manifest(out_dir, {
                "sha256": sha,
                "status": "fail",
                "http_hint": hint,
                "filename": filename,
                "attempts": attempts,
                "timestamp": ts
            })
            print(f"[FAIL] {i}/{len(remaining)} {sha} | {hint}")

        time.sleep(args.sleep)

    print(f"\nDone. ok={ok}, fail={fail}")
    print(f"Manifest: {(out_dir / 'downloads.csv').resolve()}")
    print(f"Selection: {(out_dir / 'selected_sha256.txt').resolve()}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
