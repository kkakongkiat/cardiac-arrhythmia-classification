from pathlib import Path
import sys
from time import perf_counter, sleep

# รองรับ streamlit run program/app.py จาก project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from program.common import RAW, ARTIFACTS, CLASSES, LABEL_MAP
from program.preprocessing import beat_samples
from program.inference import predict_record, STAGES

DESCRIPTIONS = {
    'N': 'กลุ่ม N รวม normal และ beat บางประเภท เช่น bundle branch block ไม่ได้หมายความว่าหัวใจปกติทั้งหมด',
    'S': 'Supraventricular ectopic beat', 'V': 'Ventricular ectopic beat',
    'F': 'Fusion of ventricular and normal beat',
    'Q': 'กลุ่ม paced, fusion of paced and normal และ unclassifiable ไม่ใช่ระดับความไม่มั่นใจ'}


def main():
    st.set_page_config(page_title='Cortex ECG', layout='wide')
    st.title('Cortex — ECG classification')
    st.caption('Offline demonstration • ใช้ตำแหน่ง heartbeat จาก annotations')
    record = st.text_input('WFDB record path (ไม่ใส่นามสกุล)', str(RAW / '100'))
    bundle = st.text_input('Model bundle', str(ARTIFACTS / 'exported'))
    mode = st.radio('โหมดแสดงผล', ['นำเสนอ', 'เวลาจริง'], horizontal=True)
    minimum = st.slider('เวลาขั้นต่ำต่อขั้นเพื่อสาธิต (วินาที)', 0., 2., .8, .1) if mode == 'นำเสนอ' else 0.
    if mode == 'นำเสนอ':
        st.caption('หน่วงเฉพาะการนำเสนอ ไม่รวมเวลาหน่วงในเวลาประมวลผลจริง')
    identity = (record, bundle)
    if st.session_state.get('input_identity') != identity:
        st.session_state.pop('result', None)
        st.session_state['input_identity'] = identity
    if st.button('เริ่มวิเคราะห์', type='primary'):
        st.session_state.pop('result', None)
        slots = {name: st.empty() for name in STAGES}
        for name, slot in slots.items():
            slot.info(f'{name}: รอ')
        state = {'delay_seconds': 0., 'started': {}}
        def progress(name, status, seconds):
            if status == 'running':
                state['started'][name] = perf_counter()
                slots[name].info(f'{name}: กำลังทำ')
            elif status == 'complete':
                remaining = max(0., minimum - (perf_counter() - state['started'][name]))
                if remaining:
                    before = perf_counter()
                    sleep(remaining)
                    state['delay_seconds'] += perf_counter() - before
                slots[name].success(f'{name}: สำเร็จ — {seconds:.4f} วินาที')
            else:
                slots[name].error(f'{name}: ผิดพลาด')
        begin = perf_counter()
        try:
            result = predict_record(bundle, record, progress=progress)
            result['flow_seconds'] = perf_counter() - begin
            result['delay_seconds'] = state['delay_seconds']
            st.session_state['result'] = result
        except Exception as error:
            st.error(f'{type(error).__name__}: {error}')
    result = st.session_state.get('result')
    if result is None:
        return
    prediction = result['predictions']
    st.subheader('เวลาที่ใช้จริง')
    columns = st.columns(4)
    columns[0].metric('ประมวลผลรวม', f"{result['processing_seconds']:.4f} s")
    columns[1].metric('Inference', f"{result['timings'][STAGES[4]]:.4f} s")
    columns[2].metric('หน่วงเพื่อนำเสนอ', f"{result['delay_seconds']:.3f} s")
    columns[3].metric('Flow ฝั่ง server', f"{result['flow_seconds']:.3f} s")
    st.caption(f"{result['device']} • {result['model_load']} • {result['beats']} beats • "
               f"เฉลี่ย {1000 * result['timings'][STAGES[4]] / result['beats']:.3f} ms/beat")
    st.caption('Flow ฝั่ง server รวมการอัปเดต UI และเวลาหน่วง แต่ไม่ได้วัดเวลาที่ browser วาดหน้าจอเสร็จ')
    st.dataframe(pd.DataFrame([{'ขั้นตอน': k, 'วินาที': v} for k, v in result['timings'].items()]), hide_index=True)
    beat = st.selectbox('เลือก heartbeat', range(len(prediction)),
                        format_func=lambda i: f"{int(prediction.iloc[i].r_sample)} — {prediction.iloc[i].predicted_class}")
    selected = prediction.iloc[beat]
    half_window = st.slider('ช่วง ECG รอบ beat (วินาทีต่อด้าน)', 1, 10, 3)
    center = int(selected.r_sample)
    left, right = max(0, center - half_window * 360), min(len(result['signal']), center + half_window * 360)
    figure = go.Figure(go.Scatter(x=np.arange(left, right) / 360, y=result['signal'][left:right], name=result['lead']))
    figure.add_vline(x=center / 360, line_color='red')
    figure.update_layout(xaxis_title='เวลา (วินาที)', yaxis_title='ECG (mV)')
    st.plotly_chart(figure, use_container_width=True)
    st.write(f"ผลทำนาย: **{selected.predicted_class}** — {DESCRIPTIONS[selected.predicted_class]}")
    st.caption('Class scores ยังไม่ได้ทำ probability calibration')
    st.bar_chart(pd.DataFrame({'score': [selected[f'score_{c}'] for c in CLASSES]}, index=CLASSES))
    display = prediction.copy()
    if st.checkbox('แสดง true label เพื่อประเมิน'):
        try:
            samples, symbols = beat_samples(Path(record))
            truth = {int(s): CLASSES[LABEL_MAP[c]] for s, c in zip(samples, symbols)}
            display['true_class'] = display.r_sample.map(truth)
        except Exception as error:
            st.error(f'อ่าน labels ไม่สำเร็จ: {error}')
    st.dataframe(display, hide_index=True)
    st.download_button('ดาวน์โหลด predictions CSV', display.to_csv(index=False).encode('utf-8-sig'),
                       'predictions.csv', 'text/csv')


if __name__ == '__main__':
    main()
