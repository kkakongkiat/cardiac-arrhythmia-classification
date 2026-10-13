# Cardiac Arrhythmia Classification — Cortex

แผนรายละเอียดอยู่ใน [PLAN.md](PLAN.md) ผลของแต่ละขั้นบันทึกใน `dataset/processed/`, `model/artifacts/` และ `model/reports/`

## Environment

ใช้ Python 3.11 และ PyTorch ที่รองรับ CUDA สำหรับ GPU ของเครื่อง ติดตั้ง PyTorch ตาม [ตัวเลือกอย่างเป็นทางการ](https://pytorch.org/get-started/locally/) ก่อนติดตั้ง dependencies ที่เหลือ

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cortex --display-name "Python (Cortex)"
jupyter lab
```

เลือก kernel ของ environment ที่ติดตั้ง dependencies แล้วใน notebooks ช่วงเวอร์ชันใน requirements เป็นข้อกำหนดเริ่มต้น ยังไม่ได้ทดสอบ environment จริง; แต่ละ run จะบันทึกเวอร์ชันที่ใช้เมื่อผู้ใช้รัน

## ลำดับ notebooks

1. `dataset/1_Processing_data.ipynb` — ตรวจ MIT-BIH ใน `dataset/raw/mit-bih-arrhythmia-database-1.0.0/` ต้องมี `RECORDS` และไฟล์ทั้ง 48 records
2. `dataset/2_Data_split.ipynb` — แบ่งผู้ป่วย 37/5/5 คนด้วย constrained optimization; Validation/Test ต้องมี Q ≥ 500 และ F ≥ 20 beats ส่วน Train ต้องมี F/Q อย่างน้อยคลาสละ 500 beats พร้อมตารางและกราฟจาก split ที่บันทึกไว้
3. `dataset/3_Preprocessing.ipynb` — เตรียม waveform; fit mean/std จาก Train
4. `dataset/4_Feature_engineering.ipynb` — เตรียม features; fit imputer จาก Train
5. `model/6_Random_forest.ipynb` ถึง `model/9_CNN_BiLSTM.ipynb` — baseline และ tuning ต่อเงื่อนไข รวม 8 configurations; DL ฝึกครั้งละหนึ่งงาน จำนวน trials แก้ใน cell `Tuning hyperparameters` (ค่าเริ่มต้น 10)
6. `model/11_Evaluation_comparison.ipynb` — ประเมิน candidate ทั้ง 8 แบบบน Test วัด latency และเลือกผู้ชนะจาก Test Macro-F1 สูงสุด ถ้าคะแนนเท่ากันเลือกตัวที่ทำนาย 1 beat ได้เร็วกว่า
7. `model/12_Export_model.ipynb` — ส่งออกโมเดลผู้ชนะจากไฟล์ 11 พร้อม preprocessing โดยไม่ฝึกใหม่

ลำดับปัจจุบันคือ `1–4 → 6–9 → 11 → 12` โดยคงเลขไฟล์เดิม

ไม่แบ่ง inner validation, ไม่ใช้ cross-validation, ไม่ฝึกซ้ำหลาย seeds และไม่ refit หลัง tuning ตัวแทนแต่ละ configuration ใช้ checkpoint ที่บันทึกจาก baseline/trial เดิม

Artifacts เป็นไฟล์ต่อเนื่องระหว่าง notebooks ไม่ใช่ตัวแปรร่วมใน kernel ทุก Notebook รันซ้ำได้: ขั้นข้อมูลเขียนทับไฟล์ของขั้นนั้น ขั้นโมเดลล้างโฟลเดอร์ configuration ของตัวเองก่อนฝึกใหม่ ขั้น 11 ล้าง `model/reports/final/` และขั้น 12 ล้าง `model/artifacts/exported/` ก่อนเขียนใหม่ Tuning เริ่ม study ใหม่และบันทึก trials ลง CSV หากเปลี่ยนข้อมูลหรือโมเดลให้รันขั้นที่ตามมาใหม่เพื่อให้ผลสอดคล้องกัน โดยยังตรวจ checksum และความถูกต้องของข้อมูล

Validation ใช้ tuning, early stopping และเลือก baseline/trial ภายในแต่ละ configuration ส่วน Test ใช้เลือกผู้ชนะระหว่าง configurations ตาม workflow นี้ ดังนั้นคะแนน Test ของผู้ชนะไม่ใช่ผลประเมินอิสระหลังเลือกโมเดล และอาจสูงกว่าผลบนข้อมูลใหม่

## ตำแหน่งโค้ด

โค้ดเตรียมข้อมูล ฝึก จูน เลือกโมเดล ประเมิน และ export อยู่ใน notebook ของแต่ละขั้นโดยตรง แต่ละไฟล์มี imports, configuration, helper definitions และ workflow cells ครบ ไม่ต้องเปิด `program/` เพื่ออ่านวิธีฝึก

ใน notebooks ของแต่ละโมเดล แยก section Baseline, Tuning on Validation และ Select and save candidates ให้อ่านและนำเสนอได้ ไม่ใช้ `%run` หรือ import pipeline จาก notebook อื่น

แก้ค่าโมเดลใน cells `Baseline hyperparameters`, `Tuning hyperparameters` และ `Training controls` ที่อยู่ต้นไฟล์ แล้วรันใหม่ตั้งแต่ต้น `BASELINE_PARAMETERS` กำหนด baseline, `TUNING_SPACE` กำหนดช่วงค้นหา และ `Training controls` กำหนด epochs/rounds, patience, batch sizes และ weighting ตามชนิดโมเดล

`baseline_macro_f1` คือ Validation Macro-F1 ของ baseline ที่ใช้ค่าตั้งต้น ส่วน `validation_macro_f1` คือ Validation Macro-F1 ของ checkpoint ที่เลือกจาก baseline และ tuning trials ใน configuration เดียวกัน ถ้า baseline ดีที่สุดสองค่านี้จะเท่ากัน ในตารางไฟล์ 11 ใช้ชื่อ `baseline_validation_macro_f1` เพื่อระบุชุดข้อมูลให้ชัดเจน และใช้ `test_macro_f1` เลือกผู้ชนะ

## กราฟการฝึกและเปรียบเทียบ

ทุกโมเดลมี section `Visualize learning curves and tuning` แสดงกราฟของ baseline และ candidate ที่เลือก ทั้ง weighted/unweighted พร้อมกราฟคะแนนแต่ละ tuning trial เทียบกับ baseline และคะแนนดีที่สุดสะสม รูปบันทึกใน `model/reports/training/<model>/`

- Random Forest: Train/Validation Macro-F1 ตามจำนวนต้นไม้ โดยใช้ prefix ของ forest ที่ฝึกแล้ว ไม่ฝึกเพิ่มเพื่อวาดกราฟ Train ใช้ stratified sample สูงสุด 10,000 beats ส่วน Validation ใช้ทั้งชุด
- XGBoost: Train/Validation Macro-F1 ตาม boosting round พร้อมเส้นระบุ round ที่บันทึก checkpoint
- CNN/CNN-BiLSTM: Train/Validation loss และ Macro-F1 ตาม epoch พร้อมเส้นระบุ epoch ที่บันทึก checkpoint ค่า Train เก็บระหว่างการฝึกแต่ละ epoch ส่วน Validation วัดหลังจบ epoch; loss ของทั้งสองชุดใช้ class weights เดียวกัน

กราฟอ่านจาก artifacts เดิมได้หลังรัน Imports, Configuration, hyperparameter/control cells และ helper definitions โดยข้าม baseline/tuning workflow ไม่ฝึกใหม่เพื่อวาดกราฟ ประวัติ XGBoost/DL ที่บันทึกก่อนเพิ่มกราฟอาจมีเฉพาะ Validation Macro-F1 และ Train loss; กราฟจะแสดงเฉพาะค่าที่มี ประวัติครบทุกช่องจะได้เมื่อฝึกครั้งถัดไป

ไฟล์ 11 มี section `Visualize best model and compare all configurations` แสดงและบันทึกใน `model/reports/final/`:

ตารางหลักใน section `Compare model performance` แสดงเพียง Model, Weighting, Accuracy, Precision (macro), Recall (macro), F1 (macro) และ Inference (ms/beat) โดยหนึ่งแถวต่อ configuration และระบุผู้ชนะด้วย `(best)` บันทึกเป็น `performance_comparison.csv` ส่วนรายละเอียดทั้งหมดเก็บใน `comparison.csv`

- Confusion matrices ของผู้ชนะ ทั้งจำนวน beats และ normalized ตาม true class พร้อมตัวเลขในช่อง
- กราฟเปรียบเทียบ Validation/Test Macro-F1 ของทั้ง 8 configurations และ heatmap ของ Test F1 รายคลาส
- กราฟ inference time สำหรับ batch 1 และ batch สูงสุด 128 เป็น median ms/beat พร้อม IQR เมื่อมีข้อมูล โดยหารด้วยจำนวน beats ที่ใช้จริง และระบุ CPU/CUDA
- กราฟ Test Macro-F1 เทียบกับ inference time และ `near_best_inference.csv` ของ configurations ที่คะแนนห่างจากดีที่สุดไม่เกิน `NEAR_BEST_TOLERANCE` (เริ่มต้น `0.005`) เรียงตามเวลา batch-1 ต่อ beat

ช่วงคะแนนใกล้เคียงใช้ประกอบการเปรียบเทียบความเร็ว ไม่เปลี่ยนผู้ชนะที่เลือกจาก Test Macro-F1 สูงสุด กราฟไม่รวม preprocessing หรือเวลาโหลดโมเดล และรันซ้ำจากรายงานที่บันทึกไว้ได้โดยไม่จับเวลาใหม่

`program/` เหลือ `app.py` สำหรับ UI และ `inference.py` สำหรับโหลด bundle, preprocessing ตอนทำนาย, model architecture และ prediction เท่านั้น (พร้อม `__init__.py`) ส่วน inference คงสำเนาขั้นตอนที่จำเป็นสำหรับแอป; หากแก้สูตร features หรือ architecture ใน notebook ให้ปรับ inference ให้ตรงกัน มี static parity checks และ notebook-to-app serialization tests เตรียมไว้

## โปรแกรมนำเสนอ

หลังส่งออก bundle แล้ว ผู้ใช้สามารถเปิดด้วย:

```powershell
python -m streamlit run program/app.py
```

เลือก local record path โดยไม่ใส่นามสกุล (เช่น record `100`) และ bundle ใน `model/artifacts/exported` แล้วกด **เริ่มวิเคราะห์** โมเดลโหลดจาก artifact ที่สร้างเองในโปรเจกต์

มีโหมดเวลาจริงและนำเสนอ เวลา delay แยกจากเวลา computation; output มี ECG, class scores, true labels แบบ optional และ CSV ใช้ annotation centers และ offline filtering ไม่ใช่ R-peak detection หรือ streaming

## การตรวจสอบที่เตรียมไว้

```powershell
python -m pytest tests -q
```

ก่อนรายงานผล ให้รัน smoke test ของทั้งสี่โมเดล ตรวจ save/load predictions ตรงกัน ตรวจ preprocessing จาก Train เท่านั้น และเปรียบเทียบโหมดเวลาจริง/นำเสนอโดยใช้ input เดียวกัน

Tests ที่ตรวจฟังก์ชันใน notebooks อ่าน imports, configuration และ definitions ผ่าน AST ส่วน tests ของการประเมิน/export รัน workflow เฉพาะกับ artifacts จำลองในโฟลเดอร์ชั่วคราว ไม่ฝึกหรือประเมิน MIT-BIH และ serialization tests ใช้ข้อมูลสังเคราะห์ขนาดเล็ก
