"""Run the notebook sequentially with a saved checkpoint after every cell.

This command downloads research data and performs the calculations. It is not
part of static code validation. Prefer VS Code's cell buttons for the course.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys

import nbformat
from nbclient import NotebookClient


class LoggedNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg.get('msg_type') == 'stream':
            print(msg['content'].get('text', '')[-3000:], end='', flush=True)
        return result


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launched-by', choices=['student', 'assistant'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root / '.venv').resolve():
        parser.error('Run using the project interpreter: .venv/bin/python tools/run_pskov_notebook.py ...')
    target = root / 'examples/pskov.ipynb'
    cache = root / '.cache'
    for name in ['matplotlib', 'ipython', 'jupyter-runtime', 'notebook-backups']:
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache / 'matplotlib')
    os.environ['IPYTHONDIR'] = str(cache / 'ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(cache / 'jupyter-runtime')
    os.environ['JUPYTER_PATH'] = str(root / '.venv/share/jupyter')
    notebook = nbformat.read(target, as_version=4)
    if any(c.cell_type == 'code' and (c.get('outputs') or c.get('execution_count') is not None) for c in notebook.cells):
        backup = cache / 'notebook-backups' / ('pskov-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.ipynb')
        backup.write_bytes(target.read_bytes())
        print(f'Previous notebook preserved: {backup}', flush=True)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    progress_path = root / 'reference/pskov-execution-status.json'
    progress = dict(started_at=now(), launched_by=args.launched_by, status='running', finished_cells=0)
    notebook.metadata['execution_provenance'] = progress.copy()

    def checkpoint():
        nbformat.validate(notebook)
        temporary = target.with_suffix('.ipynb.tmp')
        nbformat.write(notebook, temporary)
        temporary.replace(target)
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2))

    client = LoggedNotebookClient(notebook, timeout=3600, kernel_name='pskov',
                            resources={'metadata': {'path': str(root)}})
    checkpoint()
    try:
        with client.setup_kernel():
            execution_count = 0
            heading = 'Начало'
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == 'markdown':
                    if cell.source.startswith('#'):
                        heading = cell.source.splitlines()[0].lstrip('# ')
                    continue
                execution_count += 1
                progress.update(cell_index=index, execution_count=execution_count,
                                section=heading, cell_started_at=now())
                checkpoint()
                print(f'[{now()}] START cell {index + 1}: {heading}', flush=True)
                client.execute_cell(cell, index, execution_count=execution_count)
                progress['finished_cells'] = execution_count
                checkpoint()
                print(f'[{now()}] DONE cell {index + 1}', flush=True)
        progress.update(status='completed', completed_at=now())
    except BaseException as exc:
        progress.update(status='failed', failed_at=now(), error_type=type(exc).__name__, error=str(exc)[-6000:])
        raise
    finally:
        notebook.metadata['execution_provenance'] = progress.copy()
        checkpoint()
    print(f'Saved executed notebook: {target}', flush=True)


if __name__ == '__main__':
    main()
