"""Fix all notebook issues comprehensively."""
import json

BACKSLASH_N = chr(92) + 'n'  # The two-character sequence \n


def fix_literal_newlines_in_cell(cell):
    """Replace literal newline chars inside Python string literals with the escape sequence."""
    new_source = []
    for line in cell['source']:
        # Count literal newlines
        nl_count = line.count(chr(10))
        if nl_count <= 1:
            # Only has the line-ending newline, OK
            new_source.append(line)
            continue

        # Has multiple newlines - likely a string with embedded newlines
        # Strategy: replace all internal newlines (not the trailing one) with \n escape
        if line.endswith(chr(10)):
            content = line[:-1]
            content = content.replace(chr(10), BACKSLASH_N)
            new_source.append(content + chr(10))
        else:
            new_source.append(line.replace(chr(10), BACKSLASH_N))

    cell['source'] = new_source


def add_def_wrapper(cell, func_name):
    """Add a def line before the indented body after os.makedirs."""
    lines = cell['source']

    # Find makedirs line
    makedirs_idx = None
    for i, line in enumerate(lines):
        if 'os.makedirs' in line:
            makedirs_idx = i
            break

    if makedirs_idx is None:
        return False

    # Check if next non-empty line is indented (needs def)
    next_code = None
    for i in range(makedirs_idx + 1, len(lines)):
        if lines[i].strip():
            next_code = i
            break

    if next_code is None or not lines[next_code].startswith('    '):
        return False  # Already fixed or no indented body

    # Insert def line before the indented body
    new_lines = lines[:makedirs_idx + 1]
    new_lines.append(chr(10))
    new_lines.append('def ' + func_name + '():' + chr(10))
    new_lines.extend(lines[next_code:])
    cell['source'] = new_lines
    return True


def re_wrap_in_def(cell, func_name):
    """For cells already dedented: re-indent body lines and add def."""
    lines = cell['source']

    makedirs_idx = None
    for i, line in enumerate(lines):
        if 'os.makedirs' in line:
            makedirs_idx = i
            break

    if makedirs_idx is None:
        return False

    # Find the call line: gif_path = func_name()
    call_idx = None
    for i in range(makedirs_idx + 1, len(lines)):
        if func_name + '()' in lines[i]:
            call_idx = i
            break

    if call_idx is None:
        return False

    new_lines = lines[:makedirs_idx + 1]
    new_lines.append(chr(10))
    new_lines.append('def ' + func_name + '():' + chr(10))

    # Re-indent lines between makedirs and call
    for i in range(makedirs_idx + 1, call_idx):
        line = lines[i]
        if line.strip() == '':
            new_lines.append(chr(10))
        else:
            new_lines.append('    ' + line)

    # Keep call and after at top level
    new_lines.extend(lines[call_idx:])
    cell['source'] = new_lines
    return True


def verify_notebook(nb, name):
    """Check all code cells compile."""
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


# =====================================================================
# NB01
# =====================================================================
print('=== NB01 ===')
nb = json.load(open('notebooks/01_JEPA_Fundamentals.ipynb', encoding='utf-8'))

# Check if cells 53, 71, 76 still need fixing
for ci, func in [(53, 'create_training_evolution_gif'),
                 (71, 'create_gradient_flow_gif'),
                 (76, 'create_masking_gif')]:
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    try:
        compile(src, '<cell>', 'exec')
    except SyntaxError:
        # Try re-wrapping
        re_wrap_in_def(cell, func)

# Fix any literal newlines in all cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        fix_literal_newlines_in_cell(cell)

verify_notebook(nb, 'NB01')
save_nb(nb, 'notebooks/01_JEPA_Fundamentals.ipynb')

# =====================================================================
# NB02
# =====================================================================
print('=== NB02 ===')
nb = json.load(open('notebooks/02_VJEPA2_Architecture.ipynb', encoding='utf-8'))

for ci, func in [(44, 'create_attention_gif'),
                 (51, 'create_init_gif')]:
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    try:
        compile(src, '<cell>', 'exec')
    except SyntaxError:
        # Check if needs def wrapper (still indented) or re-wrap (already dedented)
        lines = cell['source']
        for i, line in enumerate(lines):
            if 'os.makedirs' in line:
                next_i = i + 1
                while next_i < len(lines) and not lines[next_i].strip():
                    next_i += 1
                if next_i < len(lines) and lines[next_i].startswith('    '):
                    add_def_wrapper(cell, func)
                else:
                    re_wrap_in_def(cell, func)
                break

# Fix literal newlines in all cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        fix_literal_newlines_in_cell(cell)

verify_notebook(nb, 'NB02')
save_nb(nb, 'notebooks/02_VJEPA2_Architecture.ipynb')

# =====================================================================
# NB03
# =====================================================================
print('=== NB03 ===')
nb = json.load(open('notebooks/03_VJEPA2_AC_WorldModel.ipynb', encoding='utf-8'))

for ci, func in [(35, 'create_cem_gif'),
                 (41, 'create_tf_vs_ar_gif')]:
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    try:
        compile(src, '<cell>', 'exec')
    except SyntaxError:
        lines = cell['source']
        for i, line in enumerate(lines):
            if 'os.makedirs' in line:
                next_i = i + 1
                while next_i < len(lines) and not lines[next_i].strip():
                    next_i += 1
                if next_i < len(lines) and lines[next_i].startswith('    '):
                    add_def_wrapper(cell, func)
                else:
                    re_wrap_in_def(cell, func)
                break

# Fix literal newlines in all cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        fix_literal_newlines_in_cell(cell)

verify_notebook(nb, 'NB03')
save_nb(nb, 'notebooks/03_VJEPA2_AC_WorldModel.ipynb')

# =====================================================================
# NB04
# =====================================================================
print('=== NB04 ===')
nb = json.load(open('notebooks/04_VLA_JEPA_Integration.ipynb', encoding='utf-8'))

# Fix cosine_similarity dimension mismatch in cell 19
cell = nb['cells'][19]
for i, line in enumerate(cell['source']):
    if "vis.detach().reshape(8, -1)" in line and "Cross-Attention" in line:
        # Replace with always using vis_pooled
        cell['source'][i] = "    vis_flat = vis_pooled.detach().reshape(8, -1).numpy()" + chr(10)
        break

for ci, func in [(29, 'create_flow_matching_gif')]:
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    try:
        compile(src, '<cell>', 'exec')
    except SyntaxError:
        lines = cell['source']
        for i, line in enumerate(lines):
            if 'os.makedirs' in line:
                next_i = i + 1
                while next_i < len(lines) and not lines[next_i].strip():
                    next_i += 1
                if next_i < len(lines) and lines[next_i].startswith('    '):
                    add_def_wrapper(cell, func)
                else:
                    re_wrap_in_def(cell, func)
                break

# Fix literal newlines
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        fix_literal_newlines_in_cell(cell)

verify_notebook(nb, 'NB04')
save_nb(nb, 'notebooks/04_VLA_JEPA_Integration.ipynb')

# =====================================================================
# NB05
# =====================================================================
print('=== NB05 ===')
nb = json.load(open('notebooks/05_ICRA_Research_Gaps.ipynb', encoding='utf-8'))

for ci, func in [(30, 'create_cbf_gif')]:
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    try:
        compile(src, '<cell>', 'exec')
    except SyntaxError:
        lines = cell['source']
        for i, line in enumerate(lines):
            if 'os.makedirs' in line:
                next_i = i + 1
                while next_i < len(lines) and not lines[next_i].strip():
                    next_i += 1
                if next_i < len(lines) and lines[next_i].startswith('    '):
                    add_def_wrapper(cell, func)
                else:
                    re_wrap_in_def(cell, func)
                break

# Fix literal newlines
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        fix_literal_newlines_in_cell(cell)

verify_notebook(nb, 'NB05')
save_nb(nb, 'notebooks/05_ICRA_Research_Gaps.ipynb')

print(chr(10) + 'All notebooks processed!')
