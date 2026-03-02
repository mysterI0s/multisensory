import os
import re

def fix_bad_tuples(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # The broken pattern is:
    # def func_name(..., param = _tuple_arg, ...):
    #   val1, val2... = _tuple_arg
    # Where val1, val2 are numbers.
    
    # Actually, it's easier to just find the line `  val1, val2... = _tuple_arg` where val1 is a digit,
    # and reconstruct the original.
    
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # If we see `def func(..., param = _tuple_arg, ...):`
        if '_tuple_arg' in line and line.strip().startswith('def '):
            # The next line might be the broken unpack
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if '= _tuple_arg' in next_line:
                    # extract the left side of the assignment
                    unpack_match = re.search(r'^\s*(.*)\s*=\s*_tuple_arg', next_line)
                    if unpack_match:
                        inner_vals = unpack_match.group(1).strip()
                        # If the inner vals are just numbers/literals, it was a default argument that got mangled
                        if any(c.isdigit() for c in inner_vals) and not any(c.isalpha() for c in inner_vals):
                            # It's a mangled default argument. Restore it.
                            restored_line = line.replace('_tuple_arg', f'({inner_vals})')
                            new_lines.append(restored_line)
                            i += 2
                            print(f'Fixed in {filepath}: {restored_line.strip()}')
                            continue
        
        # Also fix the weird line continuation error in util.py
        if 'print((\'%s%2.1f%% complete, %s %s per iteration. (%s)\' \\' in line:
             line = line.replace('\\', '')
             print(f'Fixed line continuation in util.py')

        new_lines.append(line)
        i += 1

    new_content = '\n'.join(new_lines)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            fix_bad_tuples(os.path.join(root, file))

