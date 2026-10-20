"""Publication reproduction smoke tests, executed against archived Hessians."""
from pathlib import Path
import subprocess
import sys

def test_full_paper_reproduction(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "reproduce_publication.py"),
                    "--output", str(tmp_path / "reproduced"), "--verify-raw"],
                   cwd=root, check=True)
