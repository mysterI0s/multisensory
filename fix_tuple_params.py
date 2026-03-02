import os
import re

def fix_tuple_params(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find `def function_name(... (item1, item2, ...) ...):`
    # This is tricky because there could be multiple args.
    # Let's target the exact cases we saw: `def func((a, b), c):` or `def func(a, (b, c)):`
    
    # A simpler approach: line-by-line regex replacement for the most common pattern
    # Pattern: `def \w+\(.*\([a-zA-Z0-9_, ]+\).*`
    
    lines = content.split('\n')
    new_lines = []
    
    # Regex to capture the tuple part
    # We look for something like `(a, b)` or `(x, y, w, h)` inside `def func(...)`
    # This is notoriously hard to do perfectly with regex for arbitrary python code.
    # However, for `aolib` it usually looks like: `def func_name((a, b), c=1):` or `def func_name(a, (b, c)):`
    
    for line in lines:
        if line.strip().startswith('def ') and '(' in line and ')' in line:
            # Check for tuple unpacking inside the def parameters
            # Find the parameter list between the first ( and the last ) before :
            def_match = re.search(r'def\s+\w+\s*\((.*)\)\s*:', line)
            if def_match:
                params_str = def_match.group(1)
                
                # Check if there is a tuple `(...)` inside params_str
                tuple_match = re.search(r'\(\s*([^()]+)\s*\)', params_str)
                if tuple_match:
                    inner_vars = tuple_match.group(1)
                    # Replace the `(...)` with `_tuple_arg`
                    new_params_str = params_str[:tuple_match.start()] + '_tuple_arg' + params_str[tuple_match.end():]
                    new_line = line[:def_match.start(1)] + new_params_str + line[def_match.end(1):]
                    
                    # Figure out indentation of the next line (which would be inside the function)
                    indent_match = re.match(r'^(\s*)', line)
                    base_indent = indent_match.group(1) if indent_match else ''
                    
                    # We don't know the exact indentation of the function body yet, 
                    # but usually it's `base_indent + "  "` or `base_indent + "    "`
                    # We will append the unpacking line right after the def.
                    
                    new_lines.append(new_line)
                    # Use a default 2-space indent for the unpacking line (common in aolib)
                    new_lines.append(base_indent + '  ' + inner_vars + ' = _tuple_arg')
                    continue
                    
        new_lines.append(line)

    new_content = '\n'.join(new_lines)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
            print(f"Fixed tuples in {filepath}")

# Run on all files
for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            fix_tuple_params(os.path.join(root, file))

