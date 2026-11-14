"""临时诊断：fromrecords 在 pytest 环境下的中间步骤。"""
import rsnumpy as np


def test_diag_fromrecords_steps():
    obj = [[456, 'dbe', 1.2], [2, 'de', 1.3]]
    dt = np.rec._infer_struct_dtype(obj, 'col1,col2,col3')
    print('STEP1 infer:', dt)
    a = np.array(obj, dtype=dt)
    print('STEP2 array:', a.dtype)
    print('STEP2 col2 kind:', a['col2'].dtype.kind)
    r = np.rec.array(obj, names='col1,col2,col3')
    print('STEP3 rec.array:', r.dtype)
    print('STEP3 col2 kind:', r['col2'].dtype.kind)
    r2 = np.rec.fromrecords(obj, names='col1,col2,col3')
    print('STEP4 fromrecords:', r2.dtype)
    print('STEP4 col2 kind:', r2['col2'].dtype.kind)
    assert True
