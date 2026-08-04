"""Build and push the public Hugging Face Space.

Usage:
    python scripts/deploy_space.py                      # dry run, lists what would ship
    python scripts/deploy_space.py --push               # actually upload
    python scripts/deploy_space.py --repo user/name --push

## The repository is private and the Space is public

Those are two different git repositories, and the Space is world-readable the moment it
exists. So this script does not push the project with exclusions; **it publishes only what
`build_static_space.py` constructs**. The difference matters: with an exclusion list, a new
secret ships unless someone remembers to exclude it, and the default outcome of forgetting
is a leak. With a generated tree, a file does not ship until someone writes code that emits
it, and the default outcome of forgetting is a missing chart.

That replaced an explicit allowlist of repository paths and is strictly stronger, because
there is no longer any path by which a file in the working tree reaches the Space at all.

A second guard runs regardless: the built tree is scanned for anything that looks like a
credential, and the push aborts on a hit. It exists to catch the case where the builder
itself is edited carelessly, and it has no opinion about whether the file was intended.

## Static, and why

Hugging Face retired the `streamlit` Space SDK, and Gradio and Docker Spaces are not free
on `cpu-basic`. Streamlit needed a container it was never using for anything: this
dashboard computes nothing at render time. So the Space is a static site, which is free,
and `build_static_space.py` compiles the same charts from the same data.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPO = "vyask21/bellwether"

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

# Extensions worth scanning as text. The site is all text now, including its data, so the
# scan covers every byte that ships rather than most of them.
SCANNABLE = {
    ".py",
    ".toml",
    ".md",
    ".txt",
    ".cfg",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".js",
    ".json",
    ".csv",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face Space id")
    parser.add_argument("--push", action="store_true", help="upload; otherwise dry run")
    parser.add_argument("--out", default=".space", help="build directory")
    args = parser.parse_args()

    staging = ROOT / args.out
    built = _builder().build(staging)

    print(f"\nBuilt {len(built)} files into {staging.relative_to(ROOT)}:")
    total = 0
    for path in sorted(built):
        size = path.stat().st_size
        total += size
        print(f"  {str(path.relative_to(staging)):<44}{size / 1024:>9,.0f} KB")
    print(f"  {'total':<44}{total / 1024:>9,.0f} KB")

    findings = _scan(staging)
    if findings:
        print("\nABORTED. Possible credentials in the built tree:")
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


def _builder():
    """The site builder, imported on use rather than on import.

    It reaches the dashboard's dependencies, and this module's other half is a credential
    scanner that needs none of them. Importing at module scope coupled the two, so a
    checkout with no Streamlit could not import the scanner to test it. CI is exactly that
    checkout, and it failed at collection rather than on anything it was trying to test.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static_space

    return build_static_space


def _scan(staging: Path) -> list[tuple[Path, str]]:
    """Read every text file in the built tree and match it against the patterns."""
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
        print(f"\nNot authenticated ({error}).\nRun:  hf auth login")
        return 1

    print(f"\nAuthenticated as {whoami.get('name', '?')}. Creating or updating {repo}.")
    # `static` is not a preference. The Hub retired the `streamlit` SDK, and Gradio and
    # Docker Spaces require a paid subscription on free hardware.
    api.create_repo(repo_id=repo, repo_type="space", space_sdk="static", exist_ok=True)
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
