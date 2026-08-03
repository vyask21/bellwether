"""Assemble and push the public Hugging Face Space.

Usage:
    python scripts/deploy_space.py                      # dry run, lists what would ship
    python scripts/deploy_space.py --push               # actually upload
    python scripts/deploy_space.py --repo user/name --push

## The repository is private and the Space is public

Those are two different git repositories, and the Space is world-readable the moment it
exists. So this script does not push the project with exclusions; it **builds a directory
from an explicit allowlist and pushes only that**. The difference matters: with an
exclusion list, a new secret ships unless someone remembers to exclude it, and the default
outcome of forgetting is a leak. With an allowlist, a new file does not ship until someone
names it, and the default outcome of forgetting is a missing chart.

A second guard runs regardless: the staged tree is scanned for anything that looks like a
credential, and the push aborts on a hit. It exists to catch the case where the allowlist
itself is edited carelessly, and it has no opinion about whether the file was intended.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "vyask21/bellwether"

# Everything the Space contains, named one by one. Sources are repo-relative; each maps to
# its destination inside the Space. Nothing else is copied, ever.
ALLOWLIST: tuple[tuple[str, str], ...] = (
    ("dashboard/app.py", "app.py"),
    ("dashboard/loaders.py", "loaders.py"),
    ("dashboard/viz.py", "viz.py"),
    ("dashboard/requirements.txt", "requirements.txt"),
    ("dashboard/README.md", "README.md"),
    ("dashboard/.streamlit/config.toml", ".streamlit/config.toml"),
    ("snapshot/manifest.json", "snapshot/manifest.json"),
    ("docs/backtest_results.json", "docs/backtest_results.json"),
    ("docs/operator_comparison.json", "docs/operator_comparison.json"),
    ("docs/weather_ablation.json", "docs/weather_ablation.json"),
    ("docs/breach_analysis.json", "docs/breach_analysis.json"),
    ("docs/holiday_arm.json", "docs/holiday_arm.json"),
)

# Snapshot Parquet, by market, added separately because the set is data rather than code.
MARKETS = ("CISO", "ERCO", "PACE")

# Anything matching aborts the push. Deliberately blunt: a false positive costs a minute of
# reading, a false negative publishes a key to the internet.
SECRET_PATTERNS = (
    re.compile(r"EIA_API_KEY", re.I),
    re.compile(r"HF_TOKEN|HUGGINGFACE.{0,3}TOKEN", re.I),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"api[_-]?key\s*[=:]\s*['\"][A-Za-z0-9]{16,}", re.I),
)

# Extensions worth scanning as text. Parquet is binary and holds demand readings; scanning
# it for a regex would be theatre.
SCANNABLE = {".py", ".toml", ".md", ".json", ".txt", ".cfg", ".yaml", ".yml"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face Space id")
    parser.add_argument("--push", action="store_true", help="upload; otherwise dry run")
    parser.add_argument("--out", default=".space", help="staging directory")
    args = parser.parse_args()

    staging = ROOT / args.out
    if staging.exists():
        shutil.rmtree(staging)

    staged, missing = _stage(staging)
    if missing:
        print("Missing sources, not shipped:")
        for name in missing:
            print(f"  - {name}")

    print(f"\nStaged {len(staged)} files into {staging.relative_to(ROOT)}:")
    total = 0
    for path in sorted(staged):
        size = path.stat().st_size
        total += size
        print(f"  {str(path.relative_to(staging)):<44}{size / 1024:>9,.0f} KB")
    print(f"  {'total':<44}{total / 1024:>9,.0f} KB")

    findings = _scan(staging)
    if findings:
        print("\nABORTED. Possible credentials in the staged tree:")
        for path, pattern in findings:
            print(f"  {path.relative_to(staging)}: matched {pattern}")
        return 2
    print("\nSecret scan clean.")

    # A Space is public from the moment it exists, so the confirmation is worth its space.
    if not args.push:
        print(
            f"\nDry run. Nothing uploaded.\n"
            f"Re-run with --push to publish to https://huggingface.co/spaces/{args.repo}\n"
            f"Everything listed above becomes world-readable."
        )
        return 0

    return _push(staging, args.repo)


def _stage(staging: Path) -> tuple[list[Path], list[str]]:
    staged, missing = [], []
    for source, destination in ALLOWLIST:
        origin = ROOT / source
        if not origin.exists():
            missing.append(source)
            continue
        target = staging / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        staged.append(target)

    for market in MARKETS:
        for kind in ("demand", "forecasts"):
            origin = ROOT / "snapshot" / f"{kind}_{market}.parquet"
            if not origin.exists():
                missing.append(f"snapshot/{kind}_{market}.parquet")
                continue
            target = staging / "snapshot" / origin.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            staged.append(target)
    return staged, missing


def _scan(staging: Path) -> list[tuple[Path, str]]:
    """Read every text file in the staged tree and match it against the patterns."""
    findings = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCANNABLE:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append((path, pattern.pattern))
    return findings


def _push(staging: Path, repo: str) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\nhuggingface_hub is not installed. Run:\n  pip install huggingface_hub")
        return 1

    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception as error:  # noqa: BLE001 - the message is the useful part
        print(f"\nNot authenticated ({error}).\nRun:  huggingface-cli login")
        return 1

    print(f"\nAuthenticated as {whoami.get('name', '?')}. Creating or updating {repo}.")
    api.create_repo(repo_id=repo, repo_type="space", space_sdk="streamlit", exist_ok=True)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo,
        repo_type="space",
        commit_message="Publish the findings walkthrough",
    )
    print(f"Pushed. https://huggingface.co/spaces/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
