import py_compile
import glob

errors = []
for f in glob.glob('src/**/*.py', recursive=True):
    if f.endswith('__init__.py') and 'aolib' in f:
        continue
    try:
        py_compile.compile(f, doraise=True)
    except py_compile.PyCompileError as e:
        errors.append(f"Error in {f}: {e}")

with open('errors.txt', 'w', encoding='utf-8') as f:
    f.writelines(e + '\\n' for e in errors)

if not errors:
    with open('errors.txt', 'w', encoding='utf-8') as f:
        f.write('ALL GOOD\n')
