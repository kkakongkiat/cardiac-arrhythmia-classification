"""Load definitions for tests, without running notebook workflow cells."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook_definitions(relative_path):
    book = json.loads((ROOT / relative_path).read_text(encoding='utf-8'))
    namespace = {'ROOT': ROOT, '__name__': 'notebook_test_definitions'}
    constant_names = {'RAW', 'DATA', 'ARTIFACTS', 'REPORTS', 'CLASSES', 'SYMBOLS',
                      'LABEL_MAP', 'SEED', 'PREPROCESSING'}
    nodes = []
    for cell in book['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            elif isinstance(node, ast.Assign) and all(
                    isinstance(target, ast.Name) and target.id in constant_names for target in node.targets):
                nodes.append(node)
    # Only called when the user explicitly runs pytest. No training/workflow statements.
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, relative_path, 'exec'), namespace)
    return namespace
