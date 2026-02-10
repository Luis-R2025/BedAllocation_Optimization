import importlib
import traceback
import sys
from pathlib import Path

# Ensure project root (parent of this script's parent) is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    m = importlib.import_module('src.data.get_data')
    print('Imported OK; file=', getattr(m, '__file__', None))
    print('Names:', [n for n in dir(m) if not n.startswith('_')])
except Exception:
    traceback.print_exc()
