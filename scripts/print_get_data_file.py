import importlib
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

m = importlib.import_module('src.data.get_data')
print('module file:', m.__file__)
p = Path(m.__file__)
text = p.read_text(encoding='utf-8')
lines = text.splitlines()
print('\n--- File head (first 40 lines) ---')
print('\n'.join(lines[:40]))
print('\n--- File tail (last 40 lines) ---')
print('\n'.join(lines[-40:]))
