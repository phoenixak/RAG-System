"""Copy the Space files into a local clone of the Hugging Face Space repo.

    python space/deploy.py ../hf-RAG-System-Demo     # then commit and push in that folder

The Space gets the same `rag` package and sample corpus as this repo, so it always runs the tested code.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["app.py", "style.css", "requirements.txt", "README.md"]


def main(target: str) -> None:
    dst = Path(target).resolve()
    if not (dst / ".git").exists():
        sys.exit(f"{dst} is not a git clone of the Space")
    for item in dst.iterdir():  # replace everything except git metadata
        if item.name not in {".git", ".gitattributes"}:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    for name in FILES:
        shutil.copy2(ROOT / "space" / name, dst / name)
    shutil.copytree(ROOT / "src" / "rag", dst / "src" / "rag", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "data" / "sample_corpus", dst / "sample_corpus")
    print(f"copied the Space into {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "hf-RAG-System-Demo"))
