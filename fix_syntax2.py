import glob
import re

def fix_tf_import(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Move tf.disable_v2_behavior() to the next line
    if 'tensorflow.compat.v1 as tf; tf.disable_v2_behavior()' in content:
        # We find the line containing it, and split the disable to the next line.
        lines = content.split('\\n')
        new_lines = []
        for line in lines:
            if 'tensorflow.compat.v1 as tf; tf.disable_v2_behavior()' in line:
                line = line.replace('tensorflow.compat.v1 as tf; tf.disable_v2_behavior()', 'tensorflow.compat.v1 as tf')
                new_lines.append(line)
                new_lines.append('tf.disable_v2_behavior()')
            else:
                new_lines.append(line)
        content = '\\n'.join(new_lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

def fix_tuple_unpack(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        m = re.match(r'^(\s*)def\s+([A-Za-z0-9_]+)\s*\((.*)\)\s*:(.*)$', line)
        if m:
            indent = m.group(1)
            fname = m.group(2)
            args = m.group(3)
            rest = m.group(4)
            # Find a tuple pattern like `(x, y)` in args
            # Simple assumption: only one tuple parameter
            m_tuple = re.search(r'\(([^()]+)\)', args)
            if m_tuple:
                tuple_vars = m_tuple.group(1)
                new_args = args.replace(m_tuple.group(0), '_tuple_arg')
                new_lines.append(f"{indent}def {fname}({new_args}):{rest}\\n")
                new_lines.append(f"{indent}    {tuple_vars} = _tuple_arg\\n")
                continue
        new_lines.append(line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

for fpath in glob.glob('src/**/*.py', recursive=True):
    fix_tf_import(fpath)
    fix_tuple_unpack(fpath)
