import os, ast, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
count_ok = 0
count_err = 0
errors = []

for root_dir, dirs, files in os.walk(root):
    rel = os.path.relpath(root_dir, root)
    parts = rel.split(os.sep)
    if any(p.startswith('.') for p in parts if p):
        continue
    if '__pycache__' in parts or 'node_modules' in parts:
        continue
    for f in files:
        if not f.endswith('.py'):
            continue
        fp = os.path.join(root_dir, f)
        try:
            with open(fp, 'r', encoding='utf-8') as fh:
                ast.parse(fh.read())
            count_ok += 1
        except SyntaxError as e:
            count_err += 1
            errors.append((fp, str(e)))

print(f'OK: {count_ok}, ERR: {count_err}')
for fp, e in errors[:20]:
    print(f'  ERROR: {fp}: {e}')
