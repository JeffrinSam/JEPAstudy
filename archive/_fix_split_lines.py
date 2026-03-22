"""Fix source lines that are split across multiple entries due to literal newlines in strings."""
import json

NL = chr(10)
BSN = chr(92) + 'n'


def fix_split_lines_in_cell(cell):
    """Detect and merge source lines that are split because of literal newlines in Python strings.
    A line is 'split' if it contains an unterminated string when the source lines are joined."""
    lines = cell['source']
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Try to compile this line as an expression or statement
        # If it has an unterminated string, merge with next lines
        merged = line
        attempts = 0
        while attempts < 10:
            # Check if the merged content has balanced quotes
            test = merged.rstrip(NL)
            # Simple heuristic: count unescaped single and double quotes
            # A more robust check: try to compile
            try:
                compile(test + NL, '<test>', 'exec')
                break  # Compiles OK
            except SyntaxError as e:
                if 'unterminated string' in e.msg or 'EOL while scanning' in e.msg or 'unexpected EOF' in e.msg:
                    # Merge with next line
                    i += 1
                    if i >= len(lines):
                        break
                    # Remove trailing newline from current, replace with \n escape
                    merged = merged.rstrip(NL) + BSN + lines[i].lstrip()  # Preserve indentation of first line only
                    attempts += 1
                else:
                    break  # Different error, not a split line issue
        new_lines.append(merged)
        i += 1
    cell['source'] = new_lines


def verify_cell(cell, ci):
    src = ''.join(cell['source'])
    try:
        compile(src, '<cell {}>'.format(ci), 'exec')
        return True
    except SyntaxError as e:
        return False


def verify_notebook(nb, name):
    errors = []
    for ci, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        try:
            compile(src, '<cell {}>'.format(ci), 'exec')
        except SyntaxError as e:
            txt = repr(e.text)[:60] if e.text else 'N/A'
            errors.append('  Cell {}: {} at line {}: {}'.format(ci, e.msg, e.lineno, txt))
    if errors:
        print('  {} ERRORS:'.format(name))
        for e in errors:
            print(e)
        return False
    print('  {} - All cells compile OK'.format(name))
    return True


def save_nb(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


# Fix all notebooks that still have issues
for nb_name, nb_file in [('NB01', '01_JEPA_Fundamentals.ipynb'),
                          ('NB02', '02_VJEPA2_Architecture.ipynb'),
                          ('NB03', '03_VJEPA2_AC_WorldModel.ipynb'),
                          ('NB04', '04_VLA_JEPA_Integration.ipynb'),
                          ('NB05', '05_ICRA_Research_Gaps.ipynb')]:
    print('=== {} ==='.format(nb_name))
    nb = json.load(open('notebooks/' + nb_file, encoding='utf-8'))

    # Fix any cells with split lines
    for ci, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        if not verify_cell(cell, ci):
            fix_split_lines_in_cell(cell)

    if verify_notebook(nb, nb_name):
        save_nb(nb, 'notebooks/' + nb_file)
    else:
        # Still has errors - save anyway for inspection
        save_nb(nb, 'notebooks/' + nb_file)

print(NL + 'Done!')
