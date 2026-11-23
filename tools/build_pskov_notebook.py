"""Build the unexecuted teaching notebook; never run research cells."""
from pathlib import Path
import ast
import argparse
import nbformat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--replace-executed', action='store_true', help='Explicitly replace a notebook with outputs; a backup is kept.')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / 'tools/notebook_sources/pskov.py'
cells = []
for section in source.read_text().split('# %%')[1:]:
    header, body = section.split('\n', 1)
    body = body.strip()
    if '[markdown]' in header:
        text = '\n'.join(line[2:] if line.startswith('# ') else line[1:] if line == '#' else line for line in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        ast.parse(body)
        cells.append(nbformat.v4.new_code_cell(body))
notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name': 'Псков (Python 3.12)', 'language': 'python', 'name': 'pskov'}
notebook.metadata.language_info = {'name': 'python', 'version': '3.12.14'}
nbformat.validate(notebook)
target = root / 'examples/pskov.ipynb'
if target.exists():
    existing = nbformat.read(target, as_version=4)
    has_outputs = any(c.cell_type == 'code' and (c.get('execution_count') is not None or c.get('outputs')) for c in existing.cells)
    if has_outputs:
        if not args.replace_executed:
            parser.error('Notebook already contains execution results. Preserve them; use --replace-executed only deliberately.')
        from datetime import datetime, timezone
        backup_dir = root / '.cache/notebook-backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / ('pskov-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.ipynb')
        backup.write_bytes(target.read_bytes())
        print(f'Backup: {backup}')
nbformat.write(notebook, target)
print(f'{target}: {len(cells)} cells, unexecuted')
