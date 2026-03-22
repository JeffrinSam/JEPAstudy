"""Fix NB05 cells 28 and 34 - merge split string lines."""
import json

NL = chr(10)
BSN = chr(92) + 'n'  # literal backslash-n


def save_nb(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


nb = json.load(open('notebooks/05_ICRA_Research_Gaps.ipynb', encoding='utf-8'))

# === Cell 28: merge lines 59-60 ===
# Line 59: "    ax5.text(safety[i], perf[i], compute[i], name.replace('\n"
# Line 60: "', ' '), fontsize=6)\n"
# Should be: "    ax5.text(safety[i], perf[i], compute[i], name.replace('\\n', ' '), fontsize=6)\n"
cell28 = nb['cells'][28]
merged = "    ax5.text(safety[i], perf[i], compute[i], name.replace('" + BSN + "', ' '), fontsize=6)" + NL
cell28['source'] = cell28['source'][:59] + [merged] + cell28['source'][61:]

src = ''.join(cell28['source'])
try:
    compile(src, '<cell28>', 'exec')
    print('Cell 28: OK')
except SyntaxError as e:
    print('Cell 28 error: {} at line {}'.format(e.msg, e.lineno))

# === Cell 34: merge lines 103-107 (tasks list) ===
# Lines 103-107 form: tasks = ['Spatial\n(10)', 'Object\n(10)', 'Goal\n(10)', 'Long\n(10)']
cell34 = nb['cells'][34]
merged_tasks = ("tasks = ['Spatial" + BSN + "(10)', 'Object" + BSN + "(10)', 'Goal" + BSN
                + "(10)', 'Long" + BSN + "(10)']" + NL)
cell34['source'] = cell34['source'][:103] + [merged_tasks] + cell34['source'][108:]

# After removing 4 lines (103-107 -> 1 line), line numbers shift by -4
# Old line 131 -> new line 127
# Merge lines 127-128 (the f-string split)
# Line 127: "    ax.text(y + weeks/2, 0, f'{name}\n"
# Line 128: "({weeks}w)', ha='center', va='center',\n"
# Should be: "    ax.text(y + weeks/2, 0, f'{name}\\n({weeks}w)', ha='center', va='center',\n"
line_idx = None
for i, line in enumerate(cell34['source']):
    if "f'{name}" in line and line.rstrip(NL).endswith("f'{name}"):
        line_idx = i
        break

if line_idx is not None:
    # Merge this line with the next, escaping the newline inside the f-string
    current = cell34['source'][line_idx].rstrip(NL)  # "    ax.text(y + weeks/2, 0, f'{name}"
    next_line = cell34['source'][line_idx + 1]  # "({weeks}w)', ha='center', va='center',\n"
    merged = current + BSN + next_line
    cell34['source'] = cell34['source'][:line_idx] + [merged] + cell34['source'][line_idx + 2:]
    print('Merged f-string at line {}'.format(line_idx))

src = ''.join(cell34['source'])
try:
    compile(src, '<cell34>', 'exec')
    print('Cell 34: OK')
except SyntaxError as e:
    print('Cell 34 error: {} at line {}: {}'.format(e.msg, e.lineno, repr(e.text)[:60]))

save_nb(nb, 'notebooks/05_ICRA_Research_Gaps.ipynb')
print('Saved.')
