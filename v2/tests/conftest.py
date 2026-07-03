import sys
from pathlib import Path

# v2 paketi repo-kökünden bağımsız import edilebilsin (v1 src/ ile asla karışmaz).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
