"""Static ownership/parity checks; never execute notebook cells."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook_tree(relative):
    book = json.loads((ROOT / relative).read_text(encoding='utf-8'))
    return ast.parse('\n\n'.join(''.join(cell['source']) for cell in book['cells'] if cell['cell_type'] == 'code'))


def function_ast(tree, name):
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return ast.dump(node, include_attributes=False)


def test_notebooks_do_not_import_pipeline_modules():
    for folder in ['dataset', 'model']:
        for path in (ROOT / folder).glob('*.ipynb'):
            tree = notebook_tree(path.relative_to(ROOT))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').startswith('program')
                if isinstance(node, ast.Import):
                    assert all(not name.name.startswith('program') for name in node.names)


def test_inference_matches_notebook_transforms_and_architecture():
    runtime = ast.parse((ROOT / 'program/inference.py').read_text(encoding='utf-8'))
    owners = {
        'dataset/3_Preprocessing.ipynb': ['choose_lead', 'read_record', 'segment', 'rr_features'],
        'dataset/4_Feature_engineering.ipynb': ['features'],
        'model/8_CNN.ipynb': ['ECGNetwork', 'network'],
        'model/9_CNN_BiLSTM.ipynb': ['ECGNetwork', 'network'],
    }
    for path, functions in owners.items():
        tree = notebook_tree(path)
        for name in functions:
            assert function_ast(tree, name) == function_ast(runtime, name), (path, name)


def test_model_notebooks_expose_baseline_and_tuning():
    for name in ['6_Random_forest', '7_XGBoost', '8_CNN', '9_CNN_BiLSTM']:
        book = json.loads((ROOT / f'model/{name}.ipynb').read_text(encoding='utf-8'))
        headings = {''.join(c['source']).strip() for c in book['cells'] if c['cell_type'] == 'markdown'}
        assert {'## Baseline', '## Tuning on Validation', '## Select and save candidates'} <= headings


def test_program_contains_only_app_runtime():
    assert {p.name for p in (ROOT / 'program').glob('*.py')} == {'__init__.py', 'app.py', 'inference.py'}
