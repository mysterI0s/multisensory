import glob

def fix_backslash(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace literal '\\n' with actual newline
    if '\\\\n' in content:  # wait, if it's literal \n
        pass
    content = content.replace('\\\\n', '\\n')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for fpath in glob.glob('src/**/*.py', recursive=True):
    fix_backslash(fpath)
