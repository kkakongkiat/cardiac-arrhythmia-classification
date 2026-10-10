from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset/raw/mit-bih-arrhythmia-database-1.0.0'
DATA = ROOT / 'dataset/processed'
ARTIFACTS = ROOT / 'model/artifacts'
REPORTS = ROOT / 'model/reports'
CLASSES = ['N', 'S', 'V', 'F', 'Q']
SYMBOLS = ['NLRej', 'AaJS', 'VE', 'F', '/fQ']
LABEL_MAP = {s: i for i, group in enumerate(SYMBOLS) for s in group}
SEED = 42
PREPROCESSING = {'version': 1, 'fs': 360, 'left': 90, 'right': 162,
                 'band': [0.5, 40], 'order': 4, 'leads': ['MLII', 'V5']}


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'ไม่พบ {path}; กรุณารัน notebook ขั้นก่อนหน้าก่อน')
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def new_stage(marker):
    # ห้ามเขียนทับผลเดิมเงียบ ๆ เพราะอาจทำให้โมเดลอ้างอิงคนละ split
    if Path(marker).exists():
        raise FileExistsError(f'มีผลขั้นนี้แล้ว: {marker}; โหลดผลเดิมแทนการรันซ้ำ')
    if (ARTIFACTS / 'selection.json').exists():
        raise RuntimeError('ล็อกโมเดลแล้ว ไม่อนุญาตให้เตรียมข้อมูลหรือ tuning ใหม่ใน experiment เดิม')


def environment():
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'wfdb', 'xgboost',
                'torch', 'optuna', 'streamlit', 'pyarrow']
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'packages': versions}
