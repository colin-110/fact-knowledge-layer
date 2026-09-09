import os
import tempfile
from pathlib import Path

# Must run before any `app.*` module is imported anywhere in the test session,
# so config.py points at a scratch DB/Chroma dir instead of the real data/.
_tmp_dir = Path(tempfile.mkdtemp(prefix="fact_layer_test_"))
os.environ["DATABASE_PATH"] = str(_tmp_dir / "test.db")
os.environ["CHROMA_DIR"] = str(_tmp_dir / "chroma")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
