import os
import re
import glob

def migrate_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        # 1. Print statement
        m_print = re.match(r'^(\s*)print(\s+.*)?$', line.rstrip('\r\n'))
        if m_print and not re.search(r'#.*print', line):
            indent = m_print.group(1)
            expr = m_print.group(2)
            if expr is None or expr.strip() == '':
                line = indent + "print()\n"
            else:
                expr = expr.strip()
                if not (expr.startswith('(') and expr.endswith(')')):
                    trailing_comma = False
                    if expr.endswith(','):
                        trailing_comma = True
                        expr = expr[:-1].strip()
                    m_err = re.match(r'^>>\s*([^,]+),\s*(.*)$', expr)
                    if m_err:
                        file_dest = m_err.group(1).strip()
                        expr = m_err.group(2).strip()
                        line = indent + "print(" + expr + ", file=" + file_dest + ")\n"
                    elif trailing_comma:
                        line = indent + "print(" + expr + ", end=' ')\n"
                    else:
                        line = indent + "print(" + expr + ")\n"

        # 2. xrange -> range
        line = re.sub(r'\bxrange\(', 'range(', line)

        # 3. iteritems -> items, itervalues -> values
        line = re.sub(r'\.iteritems\(\)', '.items()', line)
        line = re.sub(r'\.itervalues\(\)', '.values()', line)

        # 4. except Exception, e: -> except Exception as e:
        line = re.sub(r'except\s+([A-Za-z0-9_.]+)\s*,\s*([A-Za-z0-9_]+)\s*:', r'except \1 as \2:', line)
        
        # 5. cPickle -> pickle
        line = re.sub(r'\bcPickle\b', 'pickle', line)

        # 6. TensorFlow 1.x compatibility
        if 'tensorflow as tf' in line:
            line = re.sub(r'\btensorflow\s+as\s+tf\b', 'tensorflow.compat.v1 as tf', line)
            new_lines.append(line)
            indent_match = re.match(r'^(\s*)', line)
            import_indent = indent_match.group(1) if indent_match else ''
            new_lines.append(import_indent + "tf.disable_v2_behavior()\n")
            continue
            
        if 'tensorflow.contrib.slim as slim' in line:
            line = re.sub(r'\btensorflow\.contrib\.slim\s+as\s+slim\b', 'tf_slim as slim', line)
            
        # 7. PEP 3113: Tuple parameter unpacking
        if line.strip().startswith('def sub_img_pad(im, (x, y, w, h), oob = 0):'):
            line = line.replace('(x, y, w, h)', 'bbox')
            line = line.replace(':', ':\n    x, y, w, h = bbox')
        elif line.strip().startswith('def save_helper((fname, x)):'):
            line = line.replace('((fname, x))', '(args)')
            line = line.replace(':', ':\n    fname, x = args')
        elif line.strip().startswith('def resample_snd((snd, sr)):'):
            line = line.replace('((snd, sr))', '(args)')
            line = line.replace(':', ':\n    snd, sr = args')
        elif line.strip().startswith('def unstash_seed((py_state, np_state)):'):
            line = line.replace('((py_state, np_state))', '(args)')
            line = line.replace(':', ':\n    py_state, np_state = args')

        new_lines.append(line)
            
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        f.writelines(new_lines)

for fpath in glob.glob('src/**/*.py', recursive=True):
    migrate_file(fpath)
print("Migration done")
