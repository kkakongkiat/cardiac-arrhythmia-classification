"""Small synthetic checks. These do not run MIT-BIH preprocessing or full tuning."""
from pathlib import Path
import ast, contextlib, json, tempfile, types, sys, unittest

ROOT=Path(__file__).resolve().parents[1]

def load_definitions(relative):
    module=types.ModuleType('benchmark_test_'+Path(relative).stem)
    sys.modules[module.__name__]=module
    nb=json.loads((ROOT/relative).read_text(encoding='utf-8'))
    with contextlib.chdir(ROOT):
        for cell in nb['cells']:
            if cell['cell_type']=='code' and set(cell['metadata'].get('tags',[])) & {'definitions','configuration'}:
                exec(compile(cell['source'],str(ROOT/relative),'exec'),module.__dict__)
    return module.__dict__

class StaticChecks(unittest.TestCase):
    def test_all_notebooks_compile(self):
        import nbformat
        paths=list(ROOT.glob('dataset/*.ipynb'))+list(ROOT.glob('model/*.ipynb'))
        self.assertEqual(len(paths),9)
        for path in paths:
            nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type=='code':
                    compile(cell.source,str(path),'exec')
                    self.assertIsNone(cell.execution_count)
                    self.assertEqual(cell.outputs,[])

    def test_baseline_inside_search_spaces(self):
        settings=json.loads((ROOT/'SEARCH_SPACE.json').read_text())
        for model,baseline in settings['baselines'].items():
            space=settings['spaces'][model]
            self.assertEqual(set(space),set(baseline))
            for key,value in baseline.items():
                spec=space[key]
                if spec[0]=='categorical': self.assertIn(value,spec[1])
                else:
                    self.assertGreaterEqual(value,spec[1]); self.assertLessEqual(value,spec[2])
                    if spec[0]=='int' and len(spec)>3: self.assertEqual((value-spec[1])%spec[3],0)

    def test_architecture_matches_evaluation(self):
        reference={}
        for relative in ['model/7_CNN.ipynb','model/8_CNN_BiLSTM.ipynb','model/9_Evaluation_comparison.ipynb']:
            nb=json.loads((ROOT/relative).read_text(encoding='utf-8'))
            source='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
            classes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
                     if isinstance(n,ast.ClassDef) and n.name in ['ResidualBlock','ECGNetwork']}
            if not reference: reference=classes
            self.assertEqual(classes,reference)

    def test_training_workflow_does_not_load_test(self):
        for path in ROOT.glob('model/[5-8]_*.ipynb'):
            nb=json.loads(path.read_text(encoding='utf-8'))
            for c in nb['cells']:
                if c['cell_type']!='code': continue
                for node in ast.walk(ast.parse(c['source'])):
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='load_partition':
                        if node.args and isinstance(node.args[0],ast.Constant):
                            self.assertNotEqual(node.args[0].value,'test')

class ArrayAndProtocolChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        cls.ns=load_definitions('dataset/4_Feature_engineering.ipynb')

    def test_patient_split_and_201_202(self):
        import pandas as pd
        config=self.ns['CONFIG']
        ds1='101 106 108 109 112 114 115 116 118 119 122 124 201 202 203 205 207 208 209 215 220 223 230'.split()
        rows=[]
        for rid in ds1+config['test_records']:
            sample=0
            for label,count in enumerate([100,15,30,10]):
                for _ in range(count):
                    rows.append({'beat_id':f'{rid}:{sample}','record_id':rid,
                        'patient_group':'201_202' if rid in ['201','202'] else rid,'class_id':label})
                    sample+=1
        frame,info=self.ns['split_patients'](pd.DataFrame(rows),candidates=200)
        self.assertEqual(frame.groupby('patient_group').split.nunique().max(),1)
        self.assertEqual(set(frame.loc[frame.record_id.isin(['201','202']),'split']),{'train'} if info['record_split']['201']=='train' else {'validation'})
        self.assertEqual(set(frame.loc[frame.split.eq('test'),'record_id']),set(config['test_records']))

    def test_rr_keeps_timing_and_edge_missing(self):
        import numpy as np
        out=self.ns['rr_features'](np.array([100,460,640,1000]))
        self.assertTrue(np.isnan(out[0,0])); self.assertTrue(np.isnan(out[-1,1]))
        np.testing.assert_allclose(out[1:3,0],[1.,.5])
        self.assertEqual(out.shape,(4,12))

    def test_feature_schema_and_train_only_transforms(self):
        import numpy as np
        rng=np.random.default_rng(42)
        waves=rng.normal(size=(12,2,self.ns['WAVE_LENGTH'])).astype(np.float32)
        rr=rng.normal(size=(12,12)); rr[0,0]=np.nan
        features,names=self.ns['extract_features'](waves,rr)
        self.assertEqual(features.shape,(12,len(names)))
        part={'x':features,'rr':rr}
        transforms=self.ns['fit_transforms'](part)
        original_center=transforms['rr_scaler'].center_.copy()
        val={**part,'x':features+100,'rr':rr+100}
        converted=self.ns['transform_partition'](val,transforms)
        np.testing.assert_array_equal(transforms['rr_scaler'].center_,original_center)
        self.assertTrue(np.isfinite(converted['x']).all())
        self.assertTrue(np.isfinite(converted['context']).all())
        self.assertEqual(converted['context'].shape,(12,24))

    def test_class_weights(self):
        import numpy as np
        y=np.repeat(np.arange(4),[100,20,10,5])
        weights=self.ns['class_weights'](y,power=.5)
        self.assertGreater(weights[3],weights[0])
        self.assertAlmostEqual(float(np.mean(weights[y])),1.)
        np.testing.assert_allclose(self.ns['class_weights'](y,power=0),np.ones(4))

class ModelRoundtripChecks(unittest.TestCase):
    def test_four_models_train_save_load(self):
        import numpy as np
        import pandas as pd
        import torch
        torch.set_num_threads(2)
        rng=np.random.default_rng(42)
        def raw(n,split):
            y=np.arange(n,dtype=np.int64)%4
            return {'wave':rng.normal(size=(n,2,271)).astype(np.float32),
                    'rr':rng.normal(size=(n,12)), 'x':rng.normal(size=(n,10)).astype(np.float32),'y':y,
                    'meta':pd.DataFrame({'beat_id':[f'{split}:{i}' for i in range(n)],
                                        'class_id':y,'split':split})}
        train_raw,val_raw=raw(32,'train'),raw(16,'validation')
        for kind,path in [('rf','model/5_Random_forest.ipynb'),('xgb','model/6_XGBoost.ipynb'),
                          ('cnn','model/7_CNN.ipynb'),('cnn_bilstm','model/8_CNN_BiLSTM.ipynb')]:
            with self.subTest(model=kind),tempfile.TemporaryDirectory() as tmp:
                ns=load_definitions(path)
                ns.update({'CPU_THREADS':2,'DL_DEVICE':'cpu','XGB_DEVICE':'cpu','USE_AMP':False,
                    'MAX_EPOCHS':2,'MIN_EPOCHS':1,'EARLY_STOPPING_PATIENCE':1,
                    'MAX_BOOST_ROUNDS':10,'EARLY_STOPPING_ROUNDS':3})
                params=dict(ns['BASELINE_PARAMETERS'])
                if kind=='rf': params.update(n_estimators=8,max_depth=3)
                if kind in ['cnn','cnn_bilstm']:
                    params.update(base_channels=8,stages=2,blocks_per_stage=1,kernel=5,
                                  batch_size=16,rr_hidden=8,head_hidden=16)
                    if kind=='cnn_bilstm': params.update(lstm_hidden=8,lstm_layers=1)
                train,val,transforms=ns['prepare_training'](train_raw,val_raw)
                model,info,history=ns['fit_model'](params,train,val)
                expected=ns['predict_probabilities'](model,val)
                folder=Path(tmp)
                ns['save_model'](model,folder)
                loaded=ns['load_model'](folder,params,val['context'].shape[1])
                actual=ns['predict_probabilities'](loaded,val)
                self.assertEqual(actual.shape,(16,4))
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual.sum(1),np.ones(16),atol=1e-5)
                np.testing.assert_allclose(actual,expected,atol=1e-6)
                refit,_,_=ns['fit_model'](params,train,fixed_steps=info['best_steps'])
                self.assertEqual(ns['predict_probabilities'](refit,val).shape,(16,4))

if __name__=='__main__': unittest.main(verbosity=2)
