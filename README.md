# Cardiac Arrhythmia Classification — Cortex

โปรเจคจำแนก ECG ราย heartbeat จาก MIT-BIH เป็น 4 คลาส **N / S / V / F** ด้วย Random Forest, XGBoost, CNN และ CNN–BiLSTM

ดูวิธีใช้แอป Desktop ที่ [program/README.md](program/README.md)

## Environment

ติดตั้ง Python และ PyTorch ที่รองรับเครื่อง แล้วติดตั้ง dependencies:

```powershell
conda activate pattern
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cortex --display-name "Python (Cortex)"
jupyter lab
```

เลือก kernel ที่ติดตั้ง dependencies แล้ว และตั้ง `raw_directory` ใน `benchmark_config.json` ให้ชี้ไปยัง local MIT-BIH directory ที่มี `RECORDS` และ WFDB files

## Notebook workflow

| ลำดับ | ไฟล์ | หน้าที่ |
|---|---|---|
| 1 | `dataset/1_Processing_data.ipynb` | Audit 44 nonpacemaker records และ mapping N/S/V/F |
| 2 | `dataset/2_Data_split.ipynb` | Fixed Test cohort และ Train/Validation ที่ไม่ซ้ำผู้ป่วย |
| 3 | `dataset/3_Preprocessing.ipynb` | MLII 360 Hz, baseline removal, low-pass 35 Hz, waveform 2 × 271 |
| 4 | `dataset/4_Feature_engineering.ipynb` | Tabular features 169 ค่า และ RR context |
| 5–8 | `model/5_Random_forest.ipynb` ถึง `model/8_CNN_BiLSTM.ipynb` | Baseline, Optuna, เลือกบน Validation และ final refit |
| 9 | `model/9_Evaluation_comparison.ipynb` | ประเมิน 4 โมเดลบน Test และบันทึก checkpoint snapshots |
| 10 | `model/10_Export_model.ipynb` | Export ทั้ง 4 โมเดลพร้อม fitted transforms และ catalog เรียง Test Macro-F1 |

Hyperparameters/checkpoints เลือกด้วย Validation Macro-F1 จากนั้นค่าเริ่มต้น refit บน Train + Validation โดยใช้ epochs/rounds ที่เลือกไว้ และ fit transforms ใหม่บน development เมื่อปิด `REFIT_ON_DEVELOPMENT` จะใช้โมเดล Train ที่เลือกจาก Validation แทน

Notebook 9 เปรียบเทียบ Test Macro-F1 เพื่อจัดอันดับผู้ชนะ จึงไม่ควรใช้ Test score ของผู้ชนะเป็นการประเมินอิสระหลังเลือกโมเดล ข้อมูลข้าม notebook ส่งผ่านไฟล์ใน `dataset/processed/`, `model/artifacts/` และ `model/reports/`

## Export และ application

**ผล notebook 9 รุ่นเก่าที่ไม่มี `artifact_snapshots` ต้องรัน notebook 9 ใหม่ก่อน notebook 10** เพื่อผูกผลประเมินเข้ากับ checkpoint และ transforms ที่ใช้จริง ไม่ต้องฝึกใหม่หาก final artifacts และข้อมูลยังตรงกัน

Notebook 10 ตรวจ hash ของโมเดลและ preprocessing แล้วสร้าง `model/artifacts/exported/` เป็น catalog พร้อม bundle version 2 แยกโฟลเดอร์ `rf/`, `xgb/`, `cnn/`, `cnn_bilstm/` แต่ละ bundle มี model, fitted transforms, feature schema, configuration, preprocessing และ evaluation snapshot แอปไม่รองรับ bundle 5 คลาสรุ่นเก่า

```powershell
conda activate pattern
python program/app.py
```

เปิดเป็นหน้าต่าง Python Desktop (Tkinter + Matplotlib) โดยตรง กด **Browse ECG** เลือก `.hea` (ต้องมี `.dat` และ `.atr` คู่กัน) และ **Browse model** เลือกโฟลเดอร์ exported เลือกโมเดลจาก **Select model** ซึ่งเรียงตาม Test Macro-F1 แล้วกด **Analyze ECG**

แอปแสดงกราฟ ECG, ผลทำนายและคำอธิบายของแต่ละ beat พร้อม scores ทั้ง 4 คลาส เลือก beat ด้วยปุ่มก่อนหน้า/ถัดไป เลข beat หรือคลิกบนกราฟ กรองตามคลาส ปรับช่วงเวลา เปิด true labels และดูตารางผลทั้งหมดได้ **Export CSV** ส่งออกทั้ง record ไม่ขึ้นกับตัวกรอง การวิเคราะห์ทำงานเบื้องหลัง; Cancel จะหยุดเมื่อจบขั้นตอนปัจจุบัน

การประมวลผลเป็น **offline** ใช้ MLII 360 Hz และตำแหน่ง heartbeat จาก `.atr` annotations ใช้ทั้ง record และ future RR context ไม่ใช่การตรวจ R-peak หรือ streaming ตำแหน่ง Q ยังใช้ใน RR timing แต่ Q อยู่นอก 4 คลาสเป้าหมาย เมื่อเปิด true labels จะแสดง Q เป็น `outside task`

`program/inference.py` เก็บเฉพาะ preprocessing, model architecture และ inference ที่แอปต้องใช้ โค้ดฝึกยังอยู่ใน notebooks หากเปลี่ยนสูตรหรือ architecture ต้องปรับทั้งสองฝั่งให้ตรงกัน

## ตรวจสอบ export / application

```powershell
python -m pip install pytest
python -m pytest tests/test_model_catalog.py tests/test_desktop_application.py tests/test_benchmark_application.py tests/test_reporting_visualizations.py -q
```

ชุดทดสอบนี้ใช้ข้อมูลสังเคราะห์ ตรวจ preprocessing parity กับ notebook, export/load/predict ทั้ง 4 โมเดล, hash rejection และ Desktop UI, background worker, cancellation และ CSV รวมถึง Q labels โดยไม่ฝึก benchmark จริง ทั้งนี้ tests รุ่นเดิมบางไฟล์ยังอ้างถึง notebook/protocol 5 คลาสก่อนปรับโครงสร้าง
