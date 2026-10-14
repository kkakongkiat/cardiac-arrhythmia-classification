"""Input validation, background analysis and CSV export for the desktop app."""
from pathlib import Path
import json
import hashlib
import math

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('N', 'S', 'V', 'F')
CLASS_DETAILS = {
    'N': ('N-class beat', 'Normal / related beats',
          'ผลเบื้องต้น: beat นี้อยู่ในกลุ่ม N ซึ่งรวมทั้ง normal และความผิดปกติของการนำไฟฟ้าบางชนิด\n\n'
          'ข้อเสนอแนะ: ทบทวน ECG รอบข้างร่วมกับอาการและประวัติผู้ป่วย ไม่สรุปว่าทั้ง record ปกติจากผลนี้ '
          'หากมีอาการใจสั่นต่อเนื่อง ควรประเมิน ECG และพิจารณาติดตามจังหวะหัวใจตามข้อบ่งชี้'),
    'S': ('Supraventricular beat', 'Supraventricular ectopic beat',
          'ผลเบื้องต้น: โมเดลจัด beat นี้อยู่ในกลุ่ม supraventricular ectopic beat\n\n'
          'ข้อเสนอแนะ: ตรวจยืนยัน ECG และประเมินความถี่ร่วมกับอาการใจสั่น ทบทวนยา สารกระตุ้น และการพักผ่อน '
          'หากมีอาการเป็นช่วง ๆ อาจพิจารณา Holter หรือ event monitor ตามดุลยพินิจแพทย์'),
    'V': ('Ventricular beat', 'Ventricular ectopic beat',
          'ผลเบื้องต้น: โมเดลจัด beat นี้อยู่ในกลุ่ม ventricular ectopic beat\n\n'
          'ข้อเสนอแนะ: ยืนยันด้วย ECG ประเมินอาการและความถี่ของ beat ผิดปกติร่วมกับประวัติโรคหัวใจ '
          'หากพบซ้ำบ่อยหรือมีอาการ ควรพิจารณาติดตามจังหวะหัวใจและประเมินโครงสร้างหัวใจตามข้อบ่งชี้'),
    'F': ('Fusion beat', 'Fusion of ventricular and normal beat',
          'ผลเบื้องต้น: โมเดลจัด beat นี้อยู่ในกลุ่ม fusion ของ ventricular และ normal beat\n\n'
          'ข้อเสนอแนะ: ทบทวนรูปร่างคลื่นและ beat ก่อน–หลังเพื่อยืนยันผลและแยกสัญญาณรบกวน '
          'ประเมินร่วมกับอาการและ ECG ฉบับเต็ม ไม่ระบุชนิดของภาวะหัวใจเต้นเร็วหรือแนวทางรักษาจาก beat เดียว'),
}

MODEL_NAMES = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'cnn': 'CNN + RR', 'cnn_bilstm': 'CNN–BiLSTM + RR'}
EXPLANATION_NOTE = 'ข้อความช่วยทบทวนตามคลาส • ไม่ใช่ข้อวินิจฉัยหรือคำสั่งรักษาเฉพาะผู้ป่วย'


def discover_models(directory):
    """Read a ranked catalog or a single legacy v2 bundle; never deserialize weights."""
    folder = Path(directory.strip().strip('"')).expanduser().resolve()
    catalog_file = folder / 'catalog.json'
    if catalog_file.is_file():
        catalog = json.loads(catalog_file.read_text(encoding='utf-8'))
        if catalog.get('version') != 1 or catalog.get('ranking_metric') != 'test_macro_f1':
            raise ValueError('ไม่รองรับ catalog หรือเกณฑ์จัดอันดับนี้')
        entries = catalog.get('models', [])
        if len(entries) != 4 or {entry['model'] for entry in entries} != set(MODEL_NAMES):
            raise ValueError('Catalog ต้องมีครบทั้ง 4 โมเดล')
        models = []
        for entry in entries:
            path = (folder / entry['path']).resolve()
            if not path.is_relative_to(folder) or path == folder:
                raise ValueError('ตำแหน่งโมเดลอยู่นอกโฟลเดอร์ catalog')
            payload = (path / 'bundle.json').read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry['bundle_hash']:
                raise ValueError('Bundle เปลี่ยนหลัง export กรุณา export ใหม่')
            manifest = json.loads(payload)
            score = float(entry['test_macro_f1'])
            if (not math.isfinite(score) or not 0 <= score <= 1 or manifest.get('version') != 2
                    or manifest.get('model') != entry['model'] or manifest.get('test_macro_f1') != score
                    or manifest.get('files', {}).get('evaluation.json') != catalog.get('evaluation_hash')):
                raise ValueError('ข้อมูลอันดับไม่ตรงกับ bundle')
            models.append({**entry, 'bundle': str(path), 'score': score})
        models.sort(key=lambda entry: (-entry['score'], entry['rank']))
        for rank, entry in enumerate(models, 1):
            if entry['rank'] != rank:
                raise ValueError('ลำดับใน catalog ไม่ตรงกับคะแนน')
            entry['label'] = f"{rank}. {MODEL_NAMES[entry['model']]}  |  Test Macro-F1 {entry['score']:.4f}"
        return models
    manifest = json.loads((folder / 'bundle.json').read_text(encoding='utf-8'))
    kind = manifest.get('model')
    if manifest.get('version') != 2 or kind not in MODEL_NAMES:
        raise ValueError('กรุณาเลือก exported หรือ bundle version 2')
    score = manifest.get('test_macro_f1')
    if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('คะแนน benchmark ไม่ถูกต้อง')
    suffix = f'Test Macro-F1 {score:.4f}' if score is not None else 'ไม่มีคะแนน benchmark ใน bundle'
    return [{'model': kind, 'bundle': str(folder), 'score': score, 'label': f'{MODEL_NAMES[kind]}  |  {suffix}'}]


class AnalysisCancelled(Exception):
    pass


def default_record():
    try:
        config = json.loads((ROOT / 'benchmark_config.json').read_text(encoding='utf-8'))
        raw = Path(config['raw_directory'])
        if not raw.is_absolute():
            raw = ROOT / raw
    except (OSError, ValueError, KeyError):
        raw = ROOT / 'dataset/raw/mit-bih-arrhythmia-database-1.0.0'
    return str(raw / '100')


def record_path(value):
    path = Path(value.strip().strip('"')).expanduser()
    if path.suffix.lower() in {'.hea', '.dat', '.atr'}:
        path = path.with_suffix('')
    return path.resolve()


def validate_inputs(record, bundle):
    if not record.strip() or not bundle.strip():
        raise ValueError('กรุณาเลือกไฟล์ ECG และโฟลเดอร์โมเดลก่อนวิเคราะห์')
    record = record_path(record)
    bundle = Path(bundle.strip().strip('"')).expanduser().resolve()
    missing = [str(record) + ext for ext in ('.hea', '.dat', '.atr')
               if not Path(str(record) + ext).is_file()]
    if missing:
        raise ValueError('ไม่พบไฟล์ WFDB ที่จำเป็น:\n' + '\n'.join(missing))
    try:
        manifest = json.loads((bundle / 'bundle.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('ไม่พบ bundle.json ที่อ่านได้ กรุณาเลือกโฟลเดอร์ exported จาก notebook 10') from error
    if not isinstance(manifest, dict) or manifest.get('version') != 2:
        raise ValueError('ต้องใช้ bundle version 2 สำหรับ N/S/V/F กรุณา export ใหม่จาก notebook 10')
    return record, bundle


def analyze(record, bundle, progress, cancelled):
    """Worker-only entry point; never accesses Tk widgets."""
    def check():
        if cancelled.is_set():
            raise AnalysisCancelled()
    check()
    from program.inference import predict_record, read_beats, LABEL_MAP

    def report(name, state, elapsed):
        check()
        progress(name, state, elapsed)

    result = predict_record(bundle, record, progress=report)
    check()
    samples, symbols = read_beats(record)
    truth = {int(s): CLASSES[LABEL_MAP[c]] if c in LABEL_MAP else 'Q (outside task)'
             for s, c in zip(samples, symbols)}
    frame = result['predictions'].copy()
    frame['time_seconds'] = frame.r_sample / result['fs']
    frame['true_class'] = frame.r_sample.map(truth)
    frame['in_benchmark_scope'] = frame.true_class.isin(CLASSES)
    result.update(predictions=frame, record=str(record), bundle=str(bundle))
    check()
    return result


def export_predictions(result, destination, include_truth=False):
    frame = result['predictions'].copy()
    if not include_truth:
        frame = frame.drop(columns=['true_class', 'in_benchmark_scope'], errors='ignore')
    frame.insert(0, 'record', result['record'])
    frame.insert(1, 'model', result['model'])
    frame.to_csv(destination, index=False, encoding='utf-8-sig')
