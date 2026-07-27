import rsplotlib.pyplot as plt
import rsnumpy as np
from rsnumpy import absolute, log10, real, sum
from scipy.optimize import minimize
from skrf.calibration.deembedding import IEEEP370_SE_NZC_2xThru
from skrf.media import CPW

import skrf as rf

# 保存当前图
def ssaver(name):
    plt.savefig(name)
    plt.clf()


plt.figure()

rf.stylely()
MSL100_raw = rf.Network('./test/test_rf/skrf/data/MSL100.s2p')
MSL200_raw = rf.Network('./test/test_rf/skrf/data/MSL200.s2p')

# Keep only the data from 1MHz to 5GHz
MSL100 = MSL100_raw['1-5000mhz']
MSL200 = MSL200_raw['1-5000mhz']

plt.title('Measured data')
MSL100.plot_s_db()
MSL200.plot_s_db()
ssaver('./test/test_rf/test_data/test1.png')

plt.figure(figsize=(10, 10))
plt.suptitle('Raw measurements')

rf.stylely()
# Load raw measurements
TL100 = rf.Network('./test/test_rf/test_data/CPWG100.s2p')
TL200 = rf.Network('./test/test_rf/test_data/CPWG200.s2p')

# plot them all
plt.subplot(2, 2, 1)
TL100.plot_s_db(0, 0)
TL200.plot_s_db(0, 0)
TL100.plot_s_db(1, 1)
TL200.plot_s_db(1, 1)
plt.subplot(2, 2, 2)
TL100.plot_s_deg(0, 0)
TL200.plot_s_deg(0, 0)
TL100.plot_s_deg(1, 1)
TL200.plot_s_deg(1, 1)
plt.subplot(2, 2, 3)
TL100.plot_s_db(1, 0)
TL200.plot_s_db(1, 0)
TL100.plot_s_db(0, 1)
TL200.plot_s_db(0, 1)
plt.subplot(2, 2, 4)
TL100.plot_s_deg(1, 0)
TL200.plot_s_deg(1, 0)
TL100.plot_s_deg(0, 1)
TL200.plot_s_deg(0, 1)
ssaver('./test/test_rf/test_data/test2.png')

# deembedding using IEEEP370 impedance corrected 2xthru method
dm = IEEEP370_SE_NZC_2xThru(dummy_2xthru=TL100, name='2xthru')
fix1 = dm.s_side1
fix1.name = 'FIX-1'
fix2 = dm.s_side2
fix2.name = 'FIX-2'
d_dut = dm.deembed(TL200)
d_dut.name = 'd_DUT'

# plot them all
plt.figure(figsize=(10, 10))
plt.suptitle('Connectors models')
plt.subplot(2, 2, 1)
fix1.plot_s_db(0, 0)
fix2.plot_s_db(0, 0)
plt.subplot(2, 2, 2)
fix1.plot_s_deg(0, 0)
fix2.plot_s_deg(0, 0)
plt.subplot(2, 2, 3)
fix1.plot_s_db(1, 0)
fix2.plot_s_db(1, 0)
plt.subplot(2, 2, 4)
fix1.plot_s_deg(1, 0)
fix2.plot_s_deg(1, 0)
ssaver('./test/test_rf/test_data/test3.png')
