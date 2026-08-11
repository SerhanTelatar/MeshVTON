import os
import sys
from pathlib import Path

# So the v2 package can be imported independently of the repo root (never mixed with v1 src/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# On a machine without the 3D dependencies the builder tests run against the stub; on Colab
# (with pyrender+smplx installed) the same tests use the REAL pipeline. Production scripts
# reject the stub via assert_real_impl() — this is only a test convenience.
try:
    import pyrender  # noqa: F401
    import smplx  # noqa: F401
except ImportError:
    os.environ.setdefault("MESHVTON2_STUB", "1")
