"""Fix remaining notebook issues."""
import json

BACKSLASH_N = chr(92) + 'n'  # The two-character sequence \n
NL = chr(10)  # actual newline


def fix_literal_newlines_in_cell(cell):
    """Replace literal newline chars inside Python string literals with escape sequence.
    Only fix lines that have MORE than 1 newline (the trailing line-ending one)."""
    new_source = []
    for line in cell['source']:
        nl_count = line.count(NL)
        if nl_count <= 1:
            new_source.append(line)
            continue
        # Has multiple newlines
        if line.endswith(NL):
            content = line[:-1]
            content = content.replace(NL, BACKSLASH_N)
            new_source.append(content + NL)
        else:
            new_source.append(line.replace(NL, BACKSLASH_N))
    cell['source'] = new_source


def add_def_wrapper(cell, func_name):
    """Add a def line before the indented body after os.makedirs."""
    lines = cell['source']
    makedirs_idx = None
    for i, line in enumerate(lines):
        if 'os.makedirs' in line:
            makedirs_idx = i
            break
    if makedirs_idx is None:
        return False

    next_code = None
    for i in range(makedirs_idx + 1, len(lines)):
        if lines[i].strip():
            next_code = i
            break

    if next_code is None or not lines[next_code].startswith('    '):
        return False

    new_lines = lines[:makedirs_idx + 1]
    new_lines.append(NL)
    new_lines.append('def ' + func_name + '():' + NL)
    new_lines.extend(lines[next_code:])
    cell['source'] = new_lines
    return True


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
    else:
        print('  {} - All cells compile OK'.format(name))
        return True


def save_nb(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


# NB01 - reload from disk, fix newlines FIRST, then add def wrappers
print('=== NB01 ===')
nb = json.load(open('notebooks/01_JEPA_Fundamentals.ipynb', encoding='utf-8'))

# First fix literal newlines in ALL cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        fix_literal_newlines_in_cell(cell)

# Now add def wrappers where needed
for ci, func in [(53, 'create_training_evolution_gif'),
                 (71, 'create_gradient_flow_gif'),
                 (76, 'create_masking_gif')]:
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    try:
        compile(src, '<cell>', 'exec')
    except SyntaxError:
        add_def_wrapper(cell, func)

verify_notebook(nb, 'NB01')
save_nb(nb, 'notebooks/01_JEPA_Fundamentals.ipynb')


# NB02
print('=== NB02 ===')
nb = json.load(open('notebooks/02_VJEPA2_Architecture.ipynb', encoding='utf-8'))

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        fix_literal_newlines_in_cell(cell)

for ci, func in [(44, 'create_attention_gif'),
                 (51, 'create_init_gif')]:
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    try:
        compile(src, '<cell>', 'exec')
    except SyntaxError:
        add_def_wrapper(cell, func)

verify_notebook(nb, 'NB02')
save_nb(nb, 'notebooks/02_VJEPA2_Architecture.ipynb')


# NB05 - cell 34
print('=== NB05 ===')
nb = json.load(open('notebooks/05_ICRA_Research_Gaps.ipynb', encoding='utf-8'))

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        fix_literal_newlines_in_cell(cell)

verify_notebook(nb, 'NB05')
save_nb(nb, 'notebooks/05_ICRA_Research_Gaps.ipynb')

print(NL + 'Done!')
