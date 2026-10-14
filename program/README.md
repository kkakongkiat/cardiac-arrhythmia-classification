# ECG Arrhythmia Demo — Desktop application

แอป Python แบบหน้าต่าง ใช้ Tkinter + Matplotlib หน้าตาอ้างอิงภาพตัวอย่าง: พื้นหลังอ่อน กราฟ ECG สีชมพู และการ์ด Prediction Result / Explanation

## เปิดด้วย Anaconda env `pattern`

เปิด Anaconda Prompt ที่ project root:

```powershell
conda activate pattern
python -m pip install -r requirements.txt
python program/app.py
```

ใช้ `python -m program.app` ได้เช่นกัน ไม่ต้องเปิด browser หรือรัน Streamlit ถ้า env ไม่มี Tkinter ให้ติดตั้งด้วย `conda install -n pattern tk`

## ใช้งาน

1. **Browse ECG…** เลือก `.hea`, `.dat` หรือ `.atr` ของ record ที่ต้องการ หรือพิมพ์ base path ระบบต้องมีไฟล์ทั้งสามชนิดคู่กัน รองรับ WFDB แบบ MIT-BIH, MLII 360 Hz
2. **Browse model…** เลือกโฟลเดอร์ `model/artifacts/exported/` ที่มี `catalog.json` จาก notebook 10 หรือเลือก bundle version 2 รายโมเดล ต้องใช้ bundle ที่ export จากผล notebook 9 รุ่นที่บันทึก checkpoint snapshots
3. **Select model** เลือกโมเดลที่เรียงตาม **Test Macro-F1 จากมากไปน้อย** พร้อมคะแนนจาก notebook 9 ค่าเริ่มต้นคืออันดับหนึ่ง คะแนนนี้เป็นผล benchmark ไม่ใช่ความมั่นใจของ beat ที่กำลังดู เปลี่ยนโมเดลแล้วต้องวิเคราะห์ใหม่
4. **Analyze ECG** เริ่มงานเบื้องหลัง แถบสถานะบอกขั้นตอน หน้าต่างยังตอบสนองได้ **Cancel** ขอหยุดที่ขอบเขตขั้นตอนถัดไป ไม่ตัดการคำนวณที่กำลังทำอยู่กลางคัน
5. ดู ECG จริงของ record เลื่อน beat ด้วย **‹ / ›**, เลข beat หรือคลิกบนกราฟ เลือกช่วงแสดงผล 2/5/10/20 วินาที และกรอง predicted class ได้
6. **Show true label** แสดง annotation ของ beat เพื่อเปรียบเทียบ Q แสดงเป็น `outside task` เพราะไม่อยู่ใน 4 คลาสผลลัพธ์ แต่ตำแหน่งยังใช้คำนวณ RR
7. **All predictions** เปิดตารางทีละ 200 แถว ดับเบิลคลิกแถวเพื่อดู beat บนกราฟ **Run details** แสดงจำนวนแต่ละคลาสและเวลาของแต่ละขั้นตอน
8. **Export CSV…** บันทึกผลทั้ง record พร้อมชื่อ record, โมเดล, ตำแหน่ง/เวลา, predicted class และ class scores ถ้าเปิด Show true label จะรวม `true_class` และ `in_benchmark_scope` ด้วย

เมื่อเปลี่ยนไฟล์หรือโมเดล ผลเดิมจะถูกล้าง ต้องวิเคราะห์ใหม่ก่อนส่งออก แอปไม่บันทึกผลอัตโนมัติและไม่ส่งข้อมูลออกเครือข่าย

## ความหมายของผล

โมเดลจำแนก **ราย heartbeat** เป็น N/S/V/F การ์ด Explanation เป็นข้อเสนอแนะภาษาไทยสำหรับแพทย์ที่กำหนดตามคลาส: ผลเบื้องต้นและข้อมูลที่ควรตรวจประกอบ มี scrollbar สำหรับอ่านข้อความทั้งหมด ไม่ใช่ feature attribution การวินิจฉัย หรือคำสั่งรักษาเฉพาะผู้ป่วย N รวมทั้ง normal และ beat บางชนิดที่เกี่ยวกับ conduction จึงไม่ได้หมายความว่า ECG ทั้ง record ปกติ Class scores ยังไม่ได้ calibration

ระบบใช้ annotation centers และบริบททั้ง record รวมถึง future RR จึงเป็น offline analysis ไม่ใช่ streaming หรือ R-peak detector

## โครงสร้าง

- `app.py`: หน้าต่าง, ECG plot, navigation, ตาราง และวงจร background worker; อัปเดต widget บน main thread เท่านั้น
- `service.py`: ตรวจ input, เรียก inference, จับคู่ true labels และ export CSV
- `inference.py`: bundle validation, preprocessing, feature extraction และโมเดลทั้ง 4 แบบที่ตรงกับ notebooks

## ทดสอบ

```powershell
conda activate pattern
python -m pytest tests/test_model_catalog.py tests/test_desktop_application.py tests/test_benchmark_application.py tests/test_reporting_visualizations.py -q
```

UI tests ต้องใช้ environment ที่มี Tk รองรับการสร้างหน้าต่าง ใช้ข้อมูลสังเคราะห์ในการทดสอบ ไม่ฝึก benchmark ใหม่

## Export ทั้ง 4 โมเดล

รัน notebook 10 หลัง notebook 9 จะได้:

```text
model/artifacts/exported/
  catalog.json          # อันดับและ Test Macro-F1 จาก notebook 9
  rf/                   # bundle.json, model.joblib, transforms และ metadata
  xgb/                  # bundle.json, model.ubj, transforms และ metadata
  cnn/                  # bundle.json, model.pt, transforms และ metadata
  cnn_bilstm/           # bundle.json, model.pt, transforms และ metadata
```

Export ตรวจ hash ของโมเดลทุกตัวก่อนแทนที่ชุดเดิม หากมีตัวใดไม่ตรงผลประเมิน จะเก็บชุดเดิมไว้ ปุ่ม **Refresh models** อ่าน catalog ใหม่หลัง export โดยไม่ต้องปิดแอป การเปิดโฟลเดอร์ bundle รุ่นเก่าที่ไม่มีคะแนนจะไม่สร้างอันดับขึ้นเอง

## ที่มาของข้อความประกอบการทบทวน

ข้อความเป็น template ของแอป ไม่ใช่ guideline ที่ผ่านการรับรองทางคลินิกสำหรับระบบนี้ ประเด็นการประเมินอาการ ECG และการติดตามจังหวะหัวใจของ ectopic beats อ้างอิง [American Heart Association: PACs and PVCs](https://www.heart.org/en/health-topics/arrhythmia/about-arrhythmia/premature-contractions-pacs-and-pvcs) และ [Mayo Clinic: PVC diagnosis](https://www.mayoclinic.org/diseases-conditions/premature-ventricular-contractions/diagnosis-treatment/drc-20376762) (ตรวจ 14 ตุลาคม 2026)

สำหรับ N/F ใช้ความหมายคลาสของ benchmark และข้อเสนอให้ตรวจ ECG ฉบับเต็ม ไม่มีการอนุมานชนิดโรคหรือความเร่งด่วนจากคลาสโดยอัตโนมัติ และไม่มีคำแนะนำเรื่องยา/ขนาดยา
