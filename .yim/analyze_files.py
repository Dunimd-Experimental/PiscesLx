import os, re

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

key_files = [
    'model/core/attention.py',
    'model/core/norms.py',
    'model/core/blocks.py',
    'model/core/embedding.py',
    'model/core/cache.py',
    'model/moe/layer.py',
    'model/moe/expert.py',
    'model/moe/gate.py',
    'model/model.py',
]

for rel_path in key_files:
    fp = os.path.join(root, rel_path)
    if not os.path.exists(fp):
        print(f'{rel_path}: FILE NOT FOUND')
        continue
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    
    total_lines = len(content.splitlines())
    
    # Count docstring lines (triple-quoted strings at module/class/function level)
    # Simple heuristic: lines inside """ ... """ or ''' ... '''
    docstring_lines = 0
    in_doc = False
    doc_delim = None
    for line in content.splitlines():
        stripped = line.strip()
        if in_doc:
            docstring_lines += 1
            if doc_delim in stripped:
                in_doc = False
        else:
            if stripped.startswith('"""') and '"""' in stripped[3:]:
                docstring_lines += 1
                if stripped.count('"""') % 2 == 0:
                    continue
                else:
                    in_doc = True
                    doc_delim = '"""'
            elif stripped.startswith('"""'):
                in_doc = True
                doc_delim = '"""'
                docstring_lines += 1
            elif stripped.startswith("'''") and "'''" in stripped[3:]:
                docstring_lines += 1
                if stripped.count("'''") % 2 == 0:
                    continue
                else:
                    in_doc = True
                    doc_delim = "'''"
            elif stripped.startswith("'''"):
                in_doc = True
                doc_delim = "'''"
                docstring_lines += 1
    
    # Count class/function definitions (real code)
    defs = len(re.findall(r'^\s*(?:class |def )', content, re.MULTILINE))
    
    # Count imports
    imports = len(re.findall(r'^\s*(?:import |from )', content, re.MULTILINE))
    
    code_lines = total_lines - docstring_lines
    
    print(f'{rel_path}:')
    print(f'  Total lines: {total_lines}')
    print(f'  Docstring lines: {docstring_lines} ({docstring_lines*100//max(total_lines,1)}%)')
    print(f'  Code lines: {code_lines} ({code_lines*100//max(total_lines,1)}%)')
    print(f'  Class/def count: {defs}')
    print(f'  Import statements: {imports}')
    print()
