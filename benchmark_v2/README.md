# MIT-BIH benchmark v2 — RF / XGBoost / CNN / CNN–BiLSTM

ชุด notebooks ใหม่สำหรับเปรียบเทียบทั้ง 4 โมเดลบนข้อมูล N/S/V/F ชุดเดียวกัน เน้นการจำแนกผู้ป่วยที่ไม่อยู่ในข้อมูลฝึก ใช้ Macro-F1 เป็นตัวจูนหลัก และรายงาน Accuracy, Macro Precision/Recall, F1 รายคลาส, Balanced Accuracy และเวลา inference ประกอบ

**Paper หลัก:** de Chazal, O’Dwyer & Reilly (2004), *Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features*, IEEE Transactions on Biomedical Engineering 51(7), 1196–1206, DOI [10.1109/TBME.2004.827359](https://doi.org/10.1109/TBME.2004.827359). อ่านฉบับเต็มจาก [คลังผลงาน Trinity College Dublin](https://www.tara.tcd.ie/bitstreams/10f1bc7f-63d3-4e50-b9c4-2fae0c985b04/download)

เลือกงานนี้เป็นหลักเพราะอธิบายการแยกข้อมูลระหว่าง recordings, การกรอง ECG, morphology, RR context และการประเมินอย่างชัดเจน ชุดนี้เป็น **paper-guided extension ไม่ใช่การทำซ้ำทุกขั้นตอน**; paper เดิมใช้ linear discriminant classifiers และ 5 กลุ่ม AAMI ขณะที่ชุดนี้ใช้ 4 โมเดลตามที่ต้องการและ 4 คลาสที่ตกลงกัน

**Protocol ที่กำหนดไว้ก่อนทดลอง**

- ตัด pacemaker records 102, 104, 107, 217 และตัด Q ออกจาก classification targets; เก็บ Q timing ไว้ตอนคำนวณ RR
- ใช้ MLII โดยตรวจจากชื่อ lead; record 114 ใช้ช่องที่ชื่อ MLII จริง ไม่สมมติว่าเป็นช่องแรก
- ใช้กลุ่ม test จาก DS2 ของ paper โดยย้าย 202 ไป development เพื่อให้อยู่กับ 201 ซึ่งเป็นผู้ป่วยคนเดียวกัน ตาม [ข้อมูล PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- Test มี 21 recordings/21 คน ส่วน development มี 23 recordings/22 คน แบ่ง development เป็น Train/Validation แบบแยกผู้ป่วยด้วย seed 42 และพิจารณาจำนวนคลาสก่อนเห็นคะแนนโมเดล
- ไม่ลด N และไม่เพิ่ม/ลดตัวอย่างใน Validation/Test
- ทุกโมเดลใช้ข้อมูล waveform และ RR ของ beats เดียวกัน; RF/XGBoost ใช้ features ที่สกัดจากข้อมูลนั้น ส่วน DL เรียน waveform และมี RR branch ร่วม
- ทุกโมเดลจูน/checkpoint ด้วย Validation แล้ว refit ด้วย Train+Validation โดยใช้จำนวน epoch/boosting round ที่เลือกไว้แล้ว ไม่มี early stopping ด้วย Test
- Notebook 9 ประเมินทั้งสี่ตัวบน Test เดียวกันและจัดอันดับผล benchmark ได้

**Processing และ features**

ใช้ median filters ประมาณ 200/600 ms ลบ baseline ตามแนวทาง paper แล้ว low-pass 35 Hz แบบ zero-phase Butterworth ซึ่งเป็นการปรับจาก FIR 12 taps ใน paper ตัดหน้าต่างรอบ annotation ตั้งแต่ −250 ถึง +500 ms รวม 271 samples

Waveform มีสองช่อง: normalized ด้วย std ของ recording และ normalized ภายใน beat เอง การ normalize แบบนี้ใช้เฉพาะสัญญาณของ recording/beat ที่กำลังประมวลผล ไม่มี label มา fit ค่า สำหรับ RR และ tabular features จะ fit imputer/scaler ด้วย Train ระหว่าง tuning และ fit ด้วย development ระหว่าง final refit

RR มี pre/post, global/local mean, normalized ratios และ local variability รวม 12 features; missing indicators เพิ่มอีก 12 ช่องสำหรับ DL context ส่วน ML ใช้ fixed-interval morphology แนว paper, normalized shape/statistics และ wavelet packet energy/entropy ซึ่งเป็นส่วนต่อยอด ใช้ db4 level 4 เพื่อให้เหมาะกับหน้าต่าง 271 samples แนวคิด wavelet packet entropy อ้างอิงเสริม [Li & Zhou (2016)](https://doi.org/10.3390/e18080285)

เป็นงาน **offline beat classification**: ใช้ annotation centers, post-RR, local context และสถิติทั้ง recording จึงยังไม่ได้วัด R-peak detector หรือ inference แบบ causal streaming

**Model settings และ Optuna**

ไม่มีช่วง hyperparameters ชุดเดียวที่เป็นมาตรฐานสูงสุดสำหรับทุก dataset ช่วงต่อไปนี้เป็นขอบเขตทางวิศวกรรมที่เลือกจากพฤติกรรมของพารามิเตอร์ในเอกสารทางการ โดยใช้ Validation ตัดสินค่าที่เหมาะกับข้อมูลนี้

| โมเดล | Baseline | ช่วงค้นหาหลัก | Budget เริ่มต้น |
|---|---|---|---:|
| RF | 800 trees, leaf 2, max_features .5, sqrt class weights | 400–1600 trees; depth None/12/20/32/48; leaf 1/2/4/8; split 2/5/10/20; feature fraction .2–1; sample fraction .65–1; gini/entropy | 80 trials |
| XGBoost | depth 5, eta .03, min_child 3, subsample .85, colsample .8, L2 5 | depth 3–9; eta .005–.15 log; child weight 1–30 log; subsample .6–1; colsample .5–1; L1 1e−8–10; L2 .1–100; gamma 0–5 | 100 trials |
| CNN + RR | residual 1D CNN, 3 stages, base 32, kernel 7, RR MLP | base 16/32/64; 2–4 stages; 1–2 residual blocks/stage; kernel 5/7/11/15; dropout .1–.5; AdamW LR 1e−4–3e−3; weight decay 1e−6–1e−2; batch 64/128/256 | 80 trials |
| CNN–BiLSTM + RR | backbone แบบเดียวกัน + bidirectional LSTM hidden 64 | backbone 2–3 stages; LSTM hidden 32/64/128; 1–2 layers; search optimizer/head settings แบบ CNN | 80 trials |

ทุกโมเดลค้น class-weight power 0/.5/1 โดยคำนวณจากข้อมูลที่ใช้ฝึกจริงและจำกัดความรุนแรงของน้ำหนัก DL ใช้ cosine LR schedule, gradient clipping, augmentation เฉพาะ Train, mixed precision เมื่อใช้ CUDA และ early stopping patience 25 ภายในเพดาน 200 epochs; XGBoost ใช้ได้ถึง 3000 rounds และ patience 150

CNN–BiLSTM ในชุดนี้เรียนลำดับภายใน beat ผ่าน LSTM ร่วมกับ RR context ไม่ได้อ้างว่าเป็น sequence-to-sequence ของหลาย heartbeats

Trial 0 เป็น baseline; trial ถัดไปใช้ TPE ซึ่งมี startup 15 trials และ median pruning หลังช่วง warmup ค่า `N_TRIALS` คือจำนวน trials ทั้ง study รวม baseline และ pruned trials ปรับ budget เพิ่มแล้วรันต่อได้ผ่าน SQLite; ปัจจัยข้อมูล/ช่วงค้นหา/controls เปลี่ยนจะสร้าง run ใหม่ โมเดลและ transforms ที่ดีที่สุดบันทึกแยกตาม data/config signature

**วิธีรัน**

1. เปิด terminal ในโฟลเดอร์ `benchmark_v2` และใช้ Python environment ที่ต้องการ ติดตั้ง `requirements.txt`; หากใช้ GPU ให้เลือก PyTorch CUDA build ตาม [ตัวเลือกทางการ](https://pytorch.org/get-started/locally/)
2. ตรวจ `raw_directory` ใน `benchmark_config.json` ให้ชี้ไป MIT-BIH เดิม ไฟล์นี้รับ absolute path หรือ relative path จาก `benchmark_v2`
3. เปิด Jupyter จาก `benchmark_v2` แล้วรัน notebooks 1 → 2 → 3 → 4
4. รัน notebooks 5, 6, 7, 8 แยกกัน โดยใช้ kernel ที่ติดตั้ง dependencies ครบ การฝึก DL บน GPU ควรทำครั้งละหนึ่งงาน
5. รัน notebook 9 หลังทั้งสี่โมเดลเสร็จ อ่าน `model/reports/final/comparison.csv`, confusion matrices, errors ราย beat, รายงานราย record และ paired patient-bootstrap differences

แก้ค่าตั้งต้นและช่วงค้นหาได้ใน cells `Baseline hyperparameters`, `Optuna search space`, `Training controls` ของแต่ละโมเดล ค่าเริ่มต้นเน้นค้นหามากกว่าชุดเดิม จึงอาจใช้เวลาหลายชั่วโมงตามฮาร์ดแวร์ โดยเฉพาะ DL บน CPU สามารถลด `N_TRIALS` สำหรับการทดลองรอบแรกแล้วเพิ่มภายหลังได้

หากเปลี่ยนขั้นเตรียมข้อมูล ให้รันขั้นที่ตามมาใหม่ ระบบตรวจ hashes และ class/schema เพื่อลดการใช้ artifacts คนละรุ่น การรันซ้ำไม่ล้างโฟลเดอร์โมเดลเดิม แต่เก็บแต่ละ run ตาม signature และข้าม final refit ถ้า checkpoint ที่เลือกยังเหมือนเดิม

**การตรวจและข้อจำกัดของชุดส่งมอบ**

ตามที่ขอ ชุดนี้ไม่ได้รัน preprocessing, tuning หรือ training เต็มชุด และยังไม่มี performance ใหม่ให้รายงาน ตรวจโครงสร้าง JSON, syntax และ baseline/search-space consistency แบบ static เท่านั้น มี tests ขนาดเล็กให้รันเองได้:

```powershell
python -m unittest discover -s tests -v
```

Tests ครอบคลุม patient split, RR/missing values, train-only transforms, architecture parity ระหว่าง training/evaluation และ train/save/load/refit ด้วยข้อมูลสังเคราะห์ขนาดเล็กของทั้งสี่โมเดล ผลจาก tests ไม่ใช่ผล MIT-BIH

ข้อมูลใหม่มี 4 classes และ waveform 2 channels จึงต้องใช้ loader/architecture จากชุด v2 เมื่อทำนายหรือทำ UI; แอปเดิมที่คาดหวัง 5 classes/1 channel ยังต้องปรับแยกก่อนใช้ checkpoint v2

เอกสารพารามิเตอร์ที่ใช้ประกอบ: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), [Optuna TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
