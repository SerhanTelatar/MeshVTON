import os
import sys
from pathlib import Path

# v2 paketi repo-kökünden bağımsız import edilebilsin (v1 src/ ile asla karışmaz).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 3D bağımlılıkları olmayan makinede builder testleri stub'la koşar; Colab'da
# (pyrender+smplx kurulu) aynı testler GERÇEK hattı kullanır. Üretim scriptleri
# assert_real_impl() ile stub'ı reddeder — bu yalnız test kolaylığıdır.
try:
    import pyrender  # noqa: F401
    import smplx  # noqa: F401
except ImportError:
    os.environ.setdefault("MESHVTON2_STUB", "1")
