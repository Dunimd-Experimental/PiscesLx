"""Test if core modules can be imported successfully (not full training)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import key modules
modules_to_test = [
    'model.core.norms',
    'model.core.embedding',
    'model.moe.gate',
    'model.moe.expert',
]

for mod_name in modules_to_test:
    try:
        __import__(mod_name)
        print(f'  OK: {mod_name}')
    except Exception as e:
        print(f'  FAIL: {mod_name}: {type(e).__name__}: {str(e)[:100]}')
