"""Native ECG desktop application: python program/app.py (or python -m program.app)."""
from pathlib import Path
import sys
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import AutoMinorLocator
from program.service import (CLASSES, CLASS_DETAILS, AnalysisCancelled, analyze,
                             default_record, validate_inputs, export_predictions,
                             discover_models, EXPLANATION_NOTE)

BG, INK, MUTED, PINK, BLUE = '#f3f7fc', '#101b47', '#7183a6', '#f43f72', '#3478f6'
COLORS = {'N': '#20857b', 'S': '#d28a17', 'V': '#e43e67', 'F': '#8b60c4'}


def label(parent, text='', size=10, color=INK, bg='white', bold=False, **kwargs):
    return tk.Label(parent, text=text, bg=bg, fg=color,
                    font=('Segoe UI', size, 'bold' if bold else 'normal'), **kwargs)


def heart(parent, size=68, bg=BG):
    canvas = tk.Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0)
    points = []
    for i in range(100):
        t = 2 * math.pi * i / 100
        x = 16 * math.sin(t)**3
        y = 13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
        points.extend([size/2+x*size/36, size/2-y*size/36])
    canvas.create_polygon(points, fill=PINK, outline='')
    canvas.create_line(*[v*size for v in (.09, .51, .30, .51, .39, .32, .49, .70,
                                        .58, .43, .65, .53, .89, .53)],
                       fill='white', width=max(2, size/22), joinstyle='round')
    return canvas


class ECGApplication:
    def __init__(self, root, analyzer=analyze):
        self.root, self.analyzer = root, analyzer
        self.result, self.table_window = None, None
        self.busy = self.closed = False
        self.events = queue.Queue()
        self.cancelled = threading.Event()
        self.filtered = []
        self.models = []
        self.completed_stages = 0
        root.title('ECG Arrhythmia Demo — Cortex')
        root.geometry('1380x960')
        root.minsize(1080, 860)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.record = tk.StringVar(value=default_record())
        self.bundle = tk.StringVar(value=str(ROOT / 'model/artifacts/exported'))
        self.model_choice = tk.StringVar()
        self.beat = tk.StringVar(value='1')
        self.filter = tk.StringVar(value='All classes')
        self.window = tk.StringVar(value='10')
        self.show_truth = tk.BooleanVar(value=False)
        self._style()
        self._build()
        self.record.trace_add('write', self._input_changed)
        self.bundle.trace_add('write', self._bundle_changed)
        self.refresh_models()
        root.bind('<Left>', lambda _: self.move_beat(-1))
        root.bind('<Right>', lambda _: self.move_beat(1))
        self.poll_id = root.after(80, self._poll)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10))
        style.configure('TButton', padding=(12, 7), background='white', foreground=INK,
                        bordercolor='#dbe5f2', lightcolor='white', darkcolor='white')
        style.map('TButton', background=[('active', '#eaf1fc')], foreground=[('disabled', '#a2adbf')])
        style.configure('Accent.TButton', background=BLUE, foreground='white', bordercolor=BLUE)
        style.map('Accent.TButton', background=[('active', '#2463d5'), ('disabled', '#bdcbe1')],
                  foreground=[('disabled', '#f4f7fb')])
        style.configure('TEntry', padding=7, fieldbackground='#f8faff', bordercolor='#dce5f2')
        style.configure('TCombobox', padding=5, fieldbackground='white')
        style.configure('TSpinbox', padding=5)
        style.configure('TCheckbutton', background='white', foreground=MUTED)
        style.configure('Horizontal.TProgressbar', background=BLUE, troughcolor='#e8eef8', bordercolor='#e8eef8')
        style.configure('Treeview', rowheight=29, background='white', fieldbackground='white', foreground=INK)
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'), background='#eaf1fc')

    def _card(self, parent, color='white', border='#dfe8f5'):
        return tk.Frame(parent, bg=color, highlightbackground=border, highlightcolor=border,
                        highlightthickness=1, bd=0)

    def _build(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=24, pady=(16, 12))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(3, weight=3, minsize=230)
        body.rowconfigure(4, weight=2, minsize=265)
        header = tk.Frame(body, bg=BG)
        header.grid(row=0, column=0, sticky='ew', pady=(0, 15))
        heart(header).pack(side='left', padx=(0, 15))
        title = tk.Frame(header, bg=BG)
        title.pack(side='left')
        label(title, 'ECG Arrhythmia Demo', 25, bg=BG, bold=True).pack(anchor='w')
        label(title, 'Cardiac Arrhythmia Classification using ECG Signals', 12, MUTED, BG).pack(anchor='w', pady=(3, 0))
        label(header, 'CORTEX  /  DESKTOP\nOffline ECG analysis', 10, MUTED, BG, justify='right').pack(side='right')
        inputs = self._card(body)
        inputs.grid(row=1, column=0, sticky='ew', pady=(0, 8))
        inputs.columnconfigure(1, weight=1)
        label(inputs, 'ECG record', bold=True).grid(row=0, column=0, padx=16, pady=(14, 6), sticky='w')
        self.record_entry = ttk.Entry(inputs, textvariable=self.record)
        self.record_entry.grid(row=0, column=1, sticky='ew', pady=(14, 6))
        self.record_button = ttk.Button(inputs, text='Browse ECG…', command=self.browse_record)
        self.record_button.grid(row=0, column=2, padx=12, pady=(14, 6))
        label(inputs, 'Model bundle', bold=True).grid(row=1, column=0, padx=16, pady=(0, 12), sticky='w')
        self.bundle_entry = ttk.Entry(inputs, textvariable=self.bundle)
        self.bundle_entry.grid(row=1, column=1, sticky='ew', pady=(0, 12))
        self.bundle_button = ttk.Button(inputs, text='Browse model…', command=self.browse_bundle)
        self.bundle_button.grid(row=1, column=2, padx=12, pady=(0, 12))
        self.start_button = ttk.Button(inputs, text='Analyze ECG', style='Accent.TButton', command=self.start)
        self.start_button.grid(row=0, column=3, padx=(0, 16), pady=(14, 6), sticky='ew')
        self.cancel_button = ttk.Button(inputs, text='Cancel', command=self.cancel, state='disabled')
        self.cancel_button.grid(row=1, column=3, padx=(0, 16), pady=(0, 12), sticky='ew')
        label(inputs, 'Select model', bold=True).grid(row=2, column=0, padx=16, pady=(0, 10), sticky='w')
        self.model_box = ttk.Combobox(inputs, textvariable=self.model_choice, state='readonly')
        self.model_box.grid(row=2, column=1, columnspan=2, sticky='ew', pady=(0, 10), padx=(0, 12))
        self.model_box.bind('<<ComboboxSelected>>', lambda _: self._input_changed())
        self.refresh_button = ttk.Button(inputs, text='Refresh models', command=self.refresh_models)
        self.refresh_button.grid(row=2, column=3, padx=(0, 16), pady=(0, 10), sticky='ew')
        status = tk.Frame(body, bg=BG)
        status.grid(row=2, column=0, sticky='ew', pady=(0, 10))
        self.status = label(status, 'Ready · เลือก ECG (.hea) และ model bundle แล้วกด Analyze ECG', 10, MUTED, BG, anchor='w')
        self.status.pack(side='left', fill='x', expand=True)
        self.progress = ttk.Progressbar(status, maximum=6, length=180)
        self.progress.pack(side='right', padx=(12, 0))
        signal_card = self._card(body)
        signal_card.grid(row=3, column=0, sticky='nsew', pady=(0, 14))
        signal_card.columnconfigure(0, weight=1)
        signal_card.rowconfigure(1, weight=1)
        top = tk.Frame(signal_card, bg='white')
        top.grid(row=0, column=0, sticky='ew', padx=16, pady=(12, 3))
        label(top, 'ECG Signal', 17, bold=True).pack(side='left')
        self.signal_meta = label(top, 'MLII · 360 Hz', 10, MUTED)
        self.signal_meta.pack(side='right')
        self.figure = Figure(figsize=(11, 2.5), dpi=100, facecolor='white')
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=.065, right=.985, top=.93, bottom=.23)
        self.canvas = FigureCanvasTkAgg(self.figure, master=signal_card)
        self.canvas.get_tk_widget().configure(height=140)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew', padx=7)
        self.canvas.get_tk_widget().bind('<Configure>', self._resize_plot, add='+')
        self.canvas.mpl_connect('button_press_event', self._plot_click)
        self._empty_plot()
        nav = tk.Frame(signal_card, bg='white')
        nav.grid(row=2, column=0, sticky='ew', padx=16, pady=(0, 12))
        label(nav, 'Class', color=MUTED).pack(side='left')
        self.filter_box = ttk.Combobox(nav, textvariable=self.filter, values=['All classes', *CLASSES], width=11, state='readonly')
        self.filter_box.pack(side='left', padx=(6, 14))
        self.filter_box.bind('<<ComboboxSelected>>', lambda _: self.apply_filter())
        self.previous = ttk.Button(nav, text='‹', width=3, command=lambda: self.move_beat(-1))
        self.previous.pack(side='left')
        self.beat_spin = ttk.Spinbox(nav, from_=1, to=1, textvariable=self.beat, width=6, command=self.select_beat)
        self.beat_spin.pack(side='left', padx=6)
        self.beat_spin.bind('<Return>', lambda _: self.select_beat())
        self.beat_spin.bind('<FocusOut>', lambda _: self.select_beat())
        self.next = ttk.Button(nav, text='›', width=3, command=lambda: self.move_beat(1))
        self.next.pack(side='left')
        self.beat_count = label(nav, '/ 0 beats', color=MUTED)
        self.beat_count.pack(side='left', padx=8)
        self.truth_check = ttk.Checkbutton(nav, text='Show true label', variable=self.show_truth, command=self.select_beat)
        self.truth_check.pack(side='right')
        self.window_box = ttk.Combobox(nav, textvariable=self.window, values=['2', '5', '10', '20'], width=3, state='readonly')
        self.window_box.pack(side='right', padx=6)
        self.window_box.bind('<<ComboboxSelected>>', lambda _: self.select_beat())
        label(nav, 'Window (s)', color=MUTED).pack(side='right', padx=(5, 0))
        bottom = tk.Frame(body, bg=BG)
        bottom.grid(row=4, column=0, sticky='nsew')
        bottom.columnconfigure(0, weight=1, uniform='cards')
        bottom.columnconfigure(1, weight=1, uniform='cards')
        bottom.rowconfigure(0, weight=1)
        prediction = self._card(bottom, '#fff2f6', '#f9cbd8')
        prediction.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        label(prediction, '▥  Prediction Result', 17, bg='#fff2f6', bold=True).pack(anchor='w', padx=18, pady=(12, 4))
        self.result_title = label(prediction, 'Ready to analyze', 21, PINK, '#fff2f6', True)
        self.result_title.pack(pady=(8, 2))
        self.result_subtitle = label(prediction, 'Select a recording to get started', 11, MUTED, '#fff2f6')
        self.result_subtitle.pack()
        prediction.bind('<Configure>', lambda e: self.result_subtitle.configure(wraplength=max(200, e.width-32)))
        self.scores = tk.Canvas(prediction, height=85, bg='#fff2f6', highlightthickness=0)
        self.scores.pack(fill='x', padx=22, pady=(8, 0))
        self.scores.bind('<Configure>', lambda _: self._draw_scores())
        label(prediction, 'Per-beat class scores · not calibrated probabilities', 9, MUTED, '#fff2f6').pack(pady=(0, 8))
        explanation = self._card(bottom, '#eef6ff', '#cbdffa')
        explanation.grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        label(explanation, '◉  Explanation', 17, bg='#eef6ff', bold=True).pack(anchor='w', padx=18, pady=(12, 8))
        advice = tk.Frame(explanation, bg='#eef6ff')
        advice.pack(fill='both', expand=True, padx=22, pady=(4, 4))
        self.explanation = tk.Text(advice, height=5, wrap='word', font=('Segoe UI', 11),
                                   bg='#eef6ff', fg='#4b648d', relief='flat', borderwidth=0,
                                   highlightthickness=0, spacing3=5, cursor='arrow')
        advice_scroll = ttk.Scrollbar(advice, orient='vertical', command=self.explanation.yview)
        advice_scroll.pack(side='right', fill='y')
        self.explanation.configure(yscrollcommand=advice_scroll.set)
        self.explanation.pack(fill='both', expand=True)
        self._set_explanation('เลือกข้อมูล ECG เพื่อดูผลทำนายและข้อเสนอแนะสำหรับแพทย์ตามคลาส')
        self.truth_label = label(explanation, '', 10, '#4b648d', '#eef6ff', anchor='w')
        self.truth_label.pack(fill='x', padx=22, pady=(0, 6))
        label(explanation, EXPLANATION_NOTE, 9, MUTED, '#eef6ff', wraplength=480,
              justify='left').pack(anchor='w', padx=22, pady=(0, 12))
        footer = tk.Frame(body, bg=BG)
        footer.grid(row=5, column=0, sticky='ew', pady=(12, 0))
        self.summary = label(footer, 'N / S / V / F · Offline, annotation-centered analysis', 10, MUTED, BG)
        self.summary.pack(side='left')
        self.export_button = ttk.Button(footer, text='Export CSV…', command=self.export_csv, state='disabled')
        self.export_button.pack(side='right')
        self.table_button = ttk.Button(footer, text='All predictions', command=self.open_table, state='disabled')
        self.table_button.pack(side='right', padx=8)
        self.details_button = ttk.Button(footer, text='Run details', command=self.show_details, state='disabled')
        self.details_button.pack(side='right')
        self._navigation(False)

    def _plot_style(self):
        self.axes.set_facecolor('#f8fbff')
        self.axes.set_xlabel('Time (seconds)', color='#50638b', fontsize=9)
        self.axes.set_ylabel('ECG (mV)', color='#50638b', fontsize=9)
        self.axes.tick_params(colors='#617497', labelsize=9)
        self.axes.grid(which='major', color='#dce6f2', linewidth=.7)
        self.axes.grid(which='minor', color='#edf2f9', linewidth=.4)
        self.axes.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.axes.yaxis.set_minor_locator(AutoMinorLocator(5))
        for spine in self.axes.spines.values():
            spine.set_color('#cbd6e6')

    def _resize_plot(self, event):
        # Reserve pixel margins for axis labels even when the window is short.
        if event.width > 150 and event.height > 80:
            self.figure.subplots_adjust(left=70/event.width, right=1-16/event.width,
                                        bottom=min(.48, 44/event.height), top=1-12/event.height)
            self.canvas.draw_idle()

    def _empty_plot(self):
        self.axes.clear()
        self._plot_style()
        self.axes.set(xlim=(0, 10), ylim=(-1.5, 1.5))
        self.axes.text(.5, .5, 'Your ECG will appear here', transform=self.axes.transAxes,
                       ha='center', va='center', color=MUTED, fontsize=13)
        self.canvas.draw_idle()

    def _navigation(self, enabled):
        for widget in [self.previous, self.next, self.beat_spin, self.truth_check]:
            widget.configure(state='normal' if enabled else 'disabled')
        for widget in [self.filter_box, self.window_box]:
            widget.configure(state='readonly' if enabled else 'disabled')

    def browse_record(self):
        path = filedialog.askopenfilename(parent=self.root, title='Select a WFDB ECG record',
                                         filetypes=[('WFDB record', '*.hea *.dat *.atr'), ('All files', '*.*')])
        if path:
            self.record.set(path)

    def browse_bundle(self):
        path = filedialog.askdirectory(parent=self.root, title='Select exported model catalog or individual bundle')
        if path:
            self.bundle.set(path)

    def _set_explanation(self, text):
        self.explanation.configure(state='normal')
        self.explanation.delete('1.0', 'end')
        self.explanation.insert('1.0', text)
        self.explanation.configure(state='disabled')
        self.explanation.yview_moveto(0)

    def _bundle_changed(self, *_):
        self._input_changed()
        self.refresh_models()

    def refresh_models(self):
        if self.busy:
            return
        self._input_changed()
        try:
            self.models = discover_models(self.bundle.get())
            self.model_box.configure(values=[entry['label'] for entry in self.models], state='readonly')
            self.model_choice.set(self.models[0]['label'])
            self.status.configure(text='Models ready · เรียงตาม Test Macro-F1 จาก notebook 9 (คะแนน benchmark)', fg=MUTED)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.models = []
            self.model_box.configure(values=[], state='disabled')
            self.model_choice.set('ไม่พบโมเดล — เลือกโฟลเดอร์ exported จาก notebook 10')
            self.status.configure(text=f'Model catalog: {error}', fg=MUTED)

    def selected_bundle(self):
        for entry in self.models:
            if entry['label'] == self.model_choice.get():
                return entry['bundle']
        return self.bundle.get()

    def _input_changed(self, *_):
        if not self.busy and self.result is not None:
            self.clear_result()
            self.status.configure(text='Input changed · กด Analyze ECG เพื่อวิเคราะห์ข้อมูลใหม่', fg=MUTED)

    def clear_result(self):
        self.result, self.filtered = None, []
        self.beat.set('1')
        self.beat_count.configure(text='/ 0 beats')
        self.result_title.configure(text='Ready to analyze', fg=PINK)
        self.result_subtitle.configure(text='Select a recording to get started')
        self._set_explanation('เลือกข้อมูล ECG เพื่อดูผลทำนายและข้อเสนอแนะสำหรับแพทย์ตามคลาส')
        self.truth_label.configure(text='')
        self.summary.configure(text='N / S / V / F · Offline, annotation-centered analysis')
        self.signal_meta.configure(text='MLII · 360 Hz')
        for widget in [self.export_button, self.table_button, self.details_button]:
            widget.configure(state='disabled')
        self._navigation(False)
        if self.table_window is not None and self.table_window.winfo_exists():
            self.table_window.destroy()
        self._empty_plot()
        self._draw_scores()

    def _set_busy(self, busy):
        self.busy = busy
        for widget in [self.record_entry, self.bundle_entry, self.record_button, self.bundle_button, self.start_button, self.refresh_button]:
            widget.configure(state='disabled' if busy else 'normal')
        self.cancel_button.configure(state='normal' if busy else 'disabled')
        self.model_box.configure(state='readonly' if not busy and self.models else 'disabled')

    def start(self):
        if self.busy:
            return
        self.clear_result()
        try:
            record, bundle = validate_inputs(self.record.get(), self.selected_bundle())
        except ValueError as error:
            self.status.configure(text='ตรวจสอบข้อมูลที่เลือกก่อนเริ่มวิเคราะห์', fg='#c3375b')
            messagebox.showerror('Cannot analyze ECG', str(error), parent=self.root)
            return
        self.cancelled = threading.Event()
        self.completed_stages = 0
        self.progress['value'] = 0
        self._set_busy(True)
        self.status.configure(text='Loading analysis engine…', fg=BLUE)
        threading.Thread(target=self._work, args=(record, bundle), daemon=True, name='ecg-analysis').start()

    def _work(self, record, bundle):
        try:
            result = self.analyzer(record, bundle, lambda *e: self.events.put(('progress', e)), self.cancelled)
            self.events.put(('cancelled', None) if self.cancelled.is_set() else ('result', result))
        except AnalysisCancelled:
            self.events.put(('cancelled', None))
        except Exception as error:
            self.events.put(('cancelled', None) if self.cancelled.is_set() else ('error', f'{type(error).__name__}: {error}'))

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == 'progress':
                    name, state, _ = payload
                    if not self.cancelled.is_set():
                        self.status.configure(text=f'Analyzing · {name}', fg=BLUE)
                    if state == 'complete':
                        self.completed_stages += 1
                        self.progress['value'] = self.completed_stages
                elif kind == 'result':
                    self._set_busy(False)
                    if self.cancelled.is_set():
                        self.status.configure(text='Analysis cancelled', fg=MUTED)
                    else:
                        self.accept_result(payload)
                elif kind == 'cancelled':
                    self._set_busy(False)
                    self.status.configure(text='Analysis cancelled · สามารถเลือกข้อมูลและเริ่มใหม่ได้', fg=MUTED)
                    self.progress['value'] = 0
                elif kind == 'error':
                    self._set_busy(False)
                    self.status.configure(text='Analysis failed · ตรวจสอบข้อมูลหรือ model bundle', fg='#c3375b')
                    messagebox.showerror('Analysis failed', payload, parent=self.root)
        except queue.Empty:
            pass
        self.poll_id = self.root.after(80, self._poll)

    def cancel(self):
        if self.busy:
            self.cancelled.set()
            self.cancel_button.configure(state='disabled')
            self.status.configure(text='Cancelling… รอให้ขั้นตอนปัจจุบันสิ้นสุด', fg=MUTED)

    def accept_result(self, result):
        self.result = result
        self.progress['value'] = 6
        self.status.configure(text=f'Analysis complete · {Path(result["record"]).name} · {result["beats"]:,} beats', fg='#20857b')
        self.summary.configure(text=f'{result["model"].upper()} · {result["device"].upper()} · '
                                    f'{result["processing_seconds"]:.2f} s · {result["beats"]:,} beats')
        self.signal_meta.configure(text=f'Record {Path(result["record"]).name}  ·  {result["lead"]}  ·  {result["fs"]} Hz')
        for widget in [self.export_button, self.table_button, self.details_button]:
            widget.configure(state='normal')
        self._navigation(True)
        self.filter.set('All classes')
        self.apply_filter()

    def apply_filter(self):
        if self.result is None:
            return
        frame = self.result['predictions']
        self.filtered = np.flatnonzero(np.ones(len(frame), dtype=bool) if self.filter.get() == 'All classes'
                                       else frame.predicted_class.eq(self.filter.get()).to_numpy()).tolist()
        self.beat_spin.configure(to=max(1, len(self.filtered)))
        self.beat.set('1')
        self.beat_count.configure(text=f'/ {len(self.filtered):,} beats')
        self.select_beat()

    def selected_row(self):
        if self.result is None or not self.filtered:
            return None
        try:
            index = int(self.beat.get())-1
        except ValueError:
            index = 0
        index = min(max(0, index), len(self.filtered)-1)
        self.beat.set(str(index+1))
        return self.result['predictions'].iloc[self.filtered[index]]

    def select_beat(self):
        row = self.selected_row()
        if row is None:
            if self.result is not None:
                self.result_title.configure(text='No matching beats', fg=MUTED)
                self.result_subtitle.configure(text='Choose another class filter')
                self._set_explanation('ไม่มีผลทำนายในคลาสที่เลือก จึงไม่มีข้อเสนอแนะเฉพาะคลาส')
                self.truth_label.configure(text='')
                self._empty_plot()
            self._draw_scores()
            return
        cls = row.predicted_class
        title, description, explanation = CLASS_DETAILS[cls]
        self.result_title.configure(text=f'{cls}  ·  {title}', fg=COLORS[cls])
        self.result_subtitle.configure(text=f'{description}  |  t = {row.time_seconds:.3f} s')
        self._set_explanation(explanation)
        self.truth_label.configure(text=f'Annotation: {row.true_class}' if self.show_truth.get() else '')
        self._draw_scores()
        fs, signal = self.result['fs'], self.result['signal']
        width = min(len(signal), int(float(self.window.get())*fs))
        left = max(0, min(int(row.r_sample)-width//2, len(signal)-width))
        right = left+width
        self.axes.clear()
        self._plot_style()
        self.axes.plot(np.arange(left, right)/fs, signal[left:right], color='#f52e62', linewidth=1.2)
        self.axes.axvline(row.r_sample/fs, color=COLORS[cls], linewidth=1, linestyle='--', alpha=.8)
        self.axes.set_xlim(left/fs, max((right-1)/fs, (left+1)/fs))
        self.canvas.draw_idle()

    def _draw_scores(self):
        self.scores.delete('all')
        row = self.selected_row()
        width = max(self.scores.winfo_width(), 100)
        for i, cls in enumerate(CLASSES):
            y = 8+i*19
            value = float(row[f'score_{cls}']) if row is not None else 0
            self.scores.create_text(0, y, text=cls, anchor='w', fill=COLORS[cls], font=('Segoe UI', 9, 'bold'))
            self.scores.create_rectangle(25, y-4, width-55, y+4, fill='#f4dfe7', outline='')
            self.scores.create_rectangle(25, y-4, 25+(width-80)*value, y+4, fill=COLORS[cls], outline='')
            self.scores.create_text(width, y, text=f'{value:.1%}' if row is not None else '—',
                                    anchor='e', fill=MUTED, font=('Segoe UI', 9))

    def move_beat(self, delta):
        if not self.filtered or isinstance(self.root.focus_get(), ttk.Entry):
            return
        self.selected_row()
        self.beat.set(str(min(max(1, int(self.beat.get())+delta), len(self.filtered))))
        self.select_beat()

    def _plot_click(self, event):
        if event.inaxes != self.axes or event.xdata is None or not self.filtered:
            return
        values = self.result['predictions'].iloc[self.filtered].time_seconds.to_numpy()
        self.beat.set(str(int(np.argmin(np.abs(values-event.xdata)))+1))
        self.select_beat()

    def export_csv(self):
        if self.result is None:
            return
        destination = filedialog.asksaveasfilename(parent=self.root, title='Export all predictions',
            defaultextension='.csv', initialfile=f'{Path(self.result["record"]).name}_predictions.csv',
            filetypes=[('CSV', '*.csv')])
        if destination:
            try:
                export_predictions(self.result, destination, self.show_truth.get())
                self.status.configure(text=f'Exported all {self.result["beats"]:,} beats · {destination}', fg='#20857b')
            except OSError as error:
                messagebox.showerror('Export failed', str(error), parent=self.root)

    def show_details(self):
        if self.result is None:
            return
        result = self.result
        counts = result['predictions'].predicted_class.value_counts()
        lines = [f'Record: {result["record"]}', f'Bundle: {result["bundle"]}',
                 f'Model: {result["model"]} / {result["device"]}', '',
                 'Class counts: ' + '   '.join(f'{c}: {counts.get(c, 0):,}' for c in CLASSES), '',
                 *[f'{name}: {seconds:.4f} s' for name, seconds in result['timings'].items()],
                 '', f'Total computation: {result["processing_seconds"]:.4f} s',
                 'Offline analysis uses annotation centers and full-record RR context.']
        messagebox.showinfo('Run details', '\n'.join(lines), parent=self.root)

    def open_table(self):
        if self.result is None:
            return
        if self.table_window is not None and self.table_window.winfo_exists():
            self.table_window.lift()
            return
        window = self.table_window = tk.Toplevel(self.root)
        window.title('All predictions — double-click to inspect a beat')
        window.geometry('1080x540')
        window.configure(bg=BG)
        frame = self.result['predictions']
        columns = ['beat', 'time_seconds', 'predicted_class', *[f'score_{c}' for c in CLASSES]]
        if self.show_truth.get():
            columns.append('true_class')
        tree = ttk.Treeview(window, columns=columns, show='headings', selectmode='browse')
        scrollbar = ttk.Scrollbar(window, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        for column in columns:
            tree.heading(column, text=column.replace('_', ' ').title())
            tree.column(column, width=105, anchor='center')
        scrollbar.pack(side='right', fill='y')
        controls = tk.Frame(window, bg=BG)
        controls.pack(side='bottom', fill='x', padx=12, pady=10)
        tree.pack(fill='both', expand=True, padx=(12, 0), pady=(12, 0))
        page = [0]
        page_label = label(controls, bg=BG)
        page_label.pack(side='left')

        def show_page(delta=0):
            page[0] = min(max(0, page[0]+delta), (len(frame)-1)//200)
            tree.delete(*tree.get_children())
            start = page[0]*200
            for i in range(start, min(start+200, len(frame))):
                row = frame.iloc[i]
                values = [i+1, f'{row.time_seconds:.3f}', row.predicted_class,
                          *[f'{row[f"score_{c}"]:.4f}' for c in CLASSES]]
                if 'true_class' in columns:
                    values.append(row.true_class)
                tree.insert('', 'end', iid=str(i), values=values)
            page_label.configure(text=f'{start+1:,}–{min(start+200, len(frame)):,} of {len(frame):,} beats')

        def inspect(_):
            if tree.selection():
                index = int(tree.selection()[0])
                self.filter.set('All classes')
                self.apply_filter()
                self.beat.set(str(index+1))
                self.select_beat()
                self.root.lift()

        ttk.Button(controls, text='Next page ›', command=lambda: show_page(1)).pack(side='right')
        ttk.Button(controls, text='‹ Previous page', command=lambda: show_page(-1)).pack(side='right', padx=8)
        tree.bind('<Double-1>', inspect)
        show_page()

    def close(self):
        self.closed = True
        self.cancelled.set()
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    root = tk.Tk()
    ECGApplication(root)
    root.mainloop()


if __name__ == '__main__':
    main()
