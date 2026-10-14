"""List Hugging Face repo files with size and sha256, to fill worker/models.lock.yaml.

uv run --with huggingface_hub python scripts/dev/hf_files.py Comfy-Org/Wan-Animate-2 --glob "*.safetensors"
uv run --with huggingface_hub python scripts/dev/hf_files.py --author Comfy-Org --search wan
"""

from __future__ import annotations

import argparse
import fnmatch
import json

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repos", nargs="*")
    parser.add_argument("--glob", default="*")
    parser.add_argument("--author")
    parser.add_argument("--search")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    if args.author:
        for model in api.list_models(author=args.author, search=args.search, limit=500):
            print(model.id)
        return
    out = []
    for repo in args.repos:
        try:
            entries = api.list_repo_tree(repo, recursive=True, expand=True)
            for e in entries:
                path = getattr(e, "path", "")
                if not hasattr(e, "size") or not fnmatch.fnmatch(path, args.glob):
                    continue
                lfs = getattr(e, "lfs", None)
                sha = lfs.sha256 if lfs is not None else None
                out.append({"repo": repo, "file": path, "size": e.size, "sha256": sha})
        except Exception as exc:  # noqa: BLE001
            out.append({"repo": repo, "error": f"{type(exc).__name__}: {exc}"})
    if args.json:
        print(json.dumps(out, indent=1))
    else:
        for row in out:
            if "error" in row:
                print(f"{row['repo']}: ERROR {row['error']}")
            else:
                print(f"{row['repo']}  {row['file']}  {row['size']}  {row['sha256']}")


if __name__ == "__main__":
    main()
