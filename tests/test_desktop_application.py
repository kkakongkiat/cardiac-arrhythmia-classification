"""Exercise native Tk widgets, worker lifecycle and result navigation."""
import threading
import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from program.app import ECGApplication
from program.service import AnalysisCancelled, validate_inputs
from tests.test_benchmark_application import record, make_bundle


def sample_result():
    frame = pd.DataFrame({'r_sample': [150, 400, 650, 900],
                          'predicted_id': [0, 1, 2, 0], 'predicted_class': ['N', 'S', 'V', 'N'],
                          'true_class': ['N', 'Q (outside task)', 'V', 'F'],
                          'in_benchmark_scope': [True, False, True, True]})
    frame['time_seconds'] = frame.r_sample / 360
    for i, cls in enumerate('NSVF'):
        frame[f'score_{cls}'] = np.where(frame.predicted_id.eq(i), .85, .05)
    return {'predictions': frame, 'signal': np.sin(np.arange(1200)/20.), 'fs': 360,
            'record': '100', 'bundle': 'exported', 'lead': 'MLII', 'model': 'rf',
            'device': 'cpu', 'processing_seconds': .25, 'beats': 4, 'timings': {'inference': .1}}


@pytest.fixture(scope='module')
def tk_root():
    # One Tcl interpreter per process, matching the real desktop application.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def ui(monkeypatch, tk_root):
    root = tk.Toplevel(tk_root)
    root.withdraw()
    app = ECGApplication(root)
    errors = []
    monkeypatch.setattr('program.app.messagebox.showerror', lambda *a, **k: errors.append(a))
    app.test_errors = errors
    try:
        yield app
    finally:
        app.close()


def wait_for(ui, condition, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ui.root.update()
        if condition():
            return
        time.sleep(.01)
    pytest.fail('Desktop event loop did not finish in time')


def test_navigation_filter_plot_and_stale_inputs(ui):
    ui.accept_result(sample_result())
    ui.move_beat(1)
    assert ui.selected_row().predicted_class == 'S'
    ui.show_truth.set(True)
    ui.select_beat()
    assert 'Q (outside task)' in ui.truth_label.cget('text')
    ui.filter.set('V')
    ui.apply_filter()
    assert ui.selected_row().r_sample == 650
    ui.filter.set('F')
    ui.apply_filter()
    assert ui.selected_row() is None
    assert ui.result_title.cget('text') == 'No matching beats'
    ui.filter.set('All classes')
    ui.apply_filter()
    ui._plot_click(SimpleNamespace(inaxes=ui.axes, xdata=900/360))
    assert ui.selected_row().r_sample == 900
    ui.beat.set('bad input')
    ui.select_beat()
    assert ui.beat.get() == '1'
    ui.open_table()
    assert ui.table_window.winfo_exists()
    ui.record.set('new-record')
    assert ui.result is None and str(ui.export_button['state']) == 'disabled'
    assert not ui.table_window.winfo_exists()


def test_background_inference_and_csv(ui, tmp_path, record, monkeypatch):
    bundle, _, _ = make_bundle(tmp_path / 'project', 'rf', record)
    ui.record.set(str(record)+'.hea')
    ui.bundle.set(str(bundle))
    ui.start()
    assert ui.busy and str(ui.start_button['state']) == 'disabled'
    wait_for(ui, lambda: not ui.busy)
    assert not ui.test_errors and ui.result is not None
    assert len(ui.result['predictions']) == 4
    destination = tmp_path/'export.csv'
    monkeypatch.setattr('program.app.filedialog.asksaveasfilename', lambda **k: str(destination))
    ui.export_csv()
    assert len(pd.read_csv(destination)) == 4


def test_cancel_worker_and_restart(ui, monkeypatch):
    monkeypatch.setattr('program.app.validate_inputs', lambda *a: ('record', 'bundle'))
    entered = threading.Event()

    def slow(record, bundle, progress, cancel):
        entered.set()
        assert cancel.wait(5)
        raise AnalysisCancelled()

    ui.analyzer = slow
    ui.start()
    wait_for(ui, entered.is_set)
    ui.cancel()
    wait_for(ui, lambda: not ui.busy)
    assert ui.result is None and not ui.test_errors
    ui.analyzer = lambda *a: sample_result()
    ui.start()
    wait_for(ui, lambda: not ui.busy)
    assert ui.result is not None


def test_worker_error_restores_controls(ui, monkeypatch):
    monkeypatch.setattr('program.app.validate_inputs', lambda *a: ('record', 'bundle'))

    def broken(*args):
        raise ValueError('Invalid model')

    ui.analyzer = broken
    ui.start()
    wait_for(ui, lambda: not ui.busy)
    assert ui.result is None and ui.test_errors
    assert str(ui.start_button['state']) == 'normal'
    assert str(ui.export_button['state']) == 'disabled'


def test_preflight_does_not_start_worker(ui):
    ui.record.set('')
    ui.start()
    assert not ui.busy and ui.test_errors
    with pytest.raises(ValueError):
        validate_inputs('missing-record', 'missing-model')
