"""Fix the remaining issues in NB01, NB02, and NB05."""
import json

NL = chr(10)
BSN = chr(92) + 'n'  # literal \n (2 chars)


def save_nb(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


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


def fix_corrupted_def_line(cell):
    """Fix lines like '\\ndef func():' -> split into blank line + def line."""
    new_source = []
    for line in cell['source']:
        if line.startswith(BSN + 'def '):
            # Split into blank line + def line
            new_source.append(NL)
            new_source.append(line[2:])  # Remove the leading \n (2 chars: \ and n)
        else:
            new_source.append(line)
    cell['source'] = new_source


# =====================================================================
# NB01
# =====================================================================
print('=== NB01 ===')
nb = json.load(open('notebooks/01_JEPA_Fundamentals.ipynb', encoding='utf-8'))

for ci in [53, 71, 76]:
    fix_corrupted_def_line(nb['cells'][ci])

verify_notebook(nb, 'NB01')
save_nb(nb, 'notebooks/01_JEPA_Fundamentals.ipynb')

# =====================================================================
# NB02
# =====================================================================
print('=== NB02 ===')
nb = json.load(open('notebooks/02_VJEPA2_Architecture.ipynb', encoding='utf-8'))

for ci in [44, 51]:
    fix_corrupted_def_line(nb['cells'][ci])

verify_notebook(nb, 'NB02')
save_nb(nb, 'notebooks/02_VJEPA2_Architecture.ipynb')

# =====================================================================
# NB05 cell 34 - tasks list with literal newlines
# =====================================================================
print('=== NB05 ===')
nb = json.load(open('notebooks/05_ICRA_Research_Gaps.ipynb', encoding='utf-8'))

cell = nb['cells'][34]
# Find and fix the problematic lines
new_source = []
for line in cell['source']:
    nl_count = line.count(NL)
    if nl_count > 1:
        # Multiple newlines - escape internal ones
        if line.endswith(NL):
            content = line[:-1]
            content = content.replace(NL, BSN)
            new_source.append(content + NL)
        else:
            new_source.append(line.replace(NL, BSN))
    else:
        new_source.append(line)
cell['source'] = new_source

verify_notebook(nb, 'NB05')
save_nb(nb, 'notebooks/05_ICRA_Research_Gaps.ipynb')

print(NL + 'Done!')
