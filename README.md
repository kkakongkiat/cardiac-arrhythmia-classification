# Cardiac Arrhythmia Classification — Cortex

แผนรายละเอียดอยู่ใน [PLAN.md](PLAN.md) โค้ดส่งมอบแบบ **ยังไม่ execute** ไม่มีโมเดลหรือคะแนนทดลองที่สร้างขึ้นแล้ว

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
2. `dataset/2_Data_split.ipynb` — แบ่งผู้ป่วยครั้งเดียวเป็น Train/Validation/Test
3. `dataset/3_Preprocessing.ipynb` — เตรียม waveform; fit mean/std จาก Train
4. `dataset/4_Feature_engineering.ipynb` — เตรียม features; fit imputer จาก Train
5. `model/5_Dummy_baseline.ipynb` — สร้างโมเดลอ้างอิง
6. `model/6_Random_forest.ipynb` ถึง `model/9_CNN_BiLSTM.ipynb` — baseline และ 12 tuning trials ต่อเงื่อนไข รวม 8 configurations; DL ฝึกครั้งละหนึ่งงาน
7. `model/10_Model_selection.ipynb` — เลือกจาก Validation และล็อก checkpoint/preprocessing
8. `model/11_Evaluation_comparison.ipynb` — เปิด Test หลังตรวจ lock และสร้างรายงาน
9. `model/12_Export_model.ipynb` — ส่งออกโมเดลผู้ชนะที่เลือกจาก Validation

ไม่แบ่ง inner validation, ไม่ใช้ cross-validation, ไม่ฝึกซ้ำหลาย seeds และไม่ refit หลัง tuning ตัวแทนแต่ละ configuration ใช้ checkpoint ที่บันทึกจาก baseline/trial เดิม

Artifacts เป็นไฟล์ต่อเนื่องระหว่าง notebooks ไม่ใช่ตัวแปรร่วมใน kernel แต่ละขั้นจะปฏิเสธการเขียนทับผลที่เสร็จแล้ว หลังล็อกโมเดลจะไม่ให้เตรียมข้อมูลหรือฝึกใหม่ใน experiment เดิม หากงานหยุดกลางทางให้ตรวจและเก็บสำรอง directory ของขั้นที่ค้างก่อนเริ่มใหม่ ไม่มีการ resume trials อัตโนมัติ

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

คำสั่งทั้งหมดด้านบนเป็นขั้นตอนสำหรับผู้ใช้ ยังไม่ได้รันในการส่งมอบครั้งนี้ ไม่มีการติดตั้ง dependencies, execute notebook, ฝึกโมเดล หรือเปิด app

ก่อนรายงานผล ให้รัน smoke test ของทั้งสี่โมเดล ตรวจ save/load predictions ตรงกัน ตรวจ preprocessing จาก Train เท่านั้น และเปรียบเทียบโหมดเวลาจริง/นำเสนอโดยใช้ input เดียวกัน
