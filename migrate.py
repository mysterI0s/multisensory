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
                line = f"{indent}print()\n"
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
                        line = f"{indent}print({expr}, file={file_dest})\n"
                    elif trailing_comma:
                        line = f"{indent}print({expr}, end=' ')\n"
                    else:
                        line = f"{indent}print({expr})\n"

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
            line = re.sub(r'\btensorflow\s+as\s+tf\b', 'tensorflow.compat.v1 as tf; tf.disable_v2_behavior()', line)
        if 'tensorflow.contrib.slim as slim' in line:
            line = re.sub(r'\btensorflow\.contrib\.slim\s+as\s+slim\b', 'tf_slim as slim', line)
            
        new_lines.append(line)
            
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        f.writelines(new_lines)

for fpath in glob.glob('src/**/*.py', recursive=True):
    migrate_file(fpath)
print("Migration done")
