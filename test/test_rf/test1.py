import warnings
import rsplotlib.pyplot as plt
# import rsnumpy as np
# from rsnumpy import absolute, log10, real, sum
# from scipy.optimize import minimize
from skrf.calibration.deembedding import IEEEP370_SE_NZC_2xThru
from skrf.media import CPW
import time
import skrf as rf

# 抑制 CPW 导体损耗物理警告：低频时趋肤深度大于金属厚度/3，属于已知物理限制
warnings.filterwarnings('ignore', message='Conductor loss calculation invalid', category=RuntimeWarning)

start_time = time.time()
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

end_time = time.time()
print(f"Time cost: {end_time - start_time} seconds")

ep_r = 4.421
tanD = 0.0167

cpw = CPW(frequency=d_dut.frequency, w=1.7e-3, s=0.5e-3, t=50e-6, h=1.55e-3,
          ep_r=ep_r, tand=tanD, rho=1.7e-8, z0_port=50., has_metal_backside=True)
l_model = cpw.line(d=100.0e-3, unit='m')
l_model.name = 'model'

# plot them all
plt.figure(figsize=(10, 10))
plt.suptitle('Comparison deembedded measurement and simulation')
plt.subplot(2, 2, 1)
d_dut.plot_s_db(0, 0)
l_model.plot_s_db(0, 0,)
plt.subplot(2, 2, 2)
d_dut.plot_s_deg(0, 0)
l_model.plot_s_deg(0, 0)
plt.subplot(2, 2, 3)
d_dut.plot_s_db(1, 0)
l_model.plot_s_db(1, 0)
plt.subplot(2, 2, 4)
d_dut.plot_s_deg(1, 0)
l_model.plot_s_deg(1, 0)
ssaver('./test/test_rf/test_data/test4.png')

# compute residuals
res = dm.deembed(TL100)
res.name = 'residuals'
res.s += 1e-15  # avoid numeric singularities

# extrapolate to dc for time step
TL100_dc = TL100.extrapolate_to_dc(kind='linear')
TL200_dc = TL200.extrapolate_to_dc(kind='linear')
fix1_dc = fix1.extrapolate_to_dc(kind='cubic')
fix2_dc = fix2.extrapolate_to_dc(kind='cubic')
d_dut_dc = d_dut.extrapolate_to_dc(kind='cubic')

# plot them all
# time domain
plt.figure(figsize=(8, 4))
plt.suptitle('Time domain reflexion step response (DC extrapolation)')
TL100_dc.plot_z_time_step(0, 0)
TL200_dc.plot_z_time_step(0, 0)
fix1_dc.plot_z_time_step(0, 0)
fix2_dc.plot_z_time_step(0, 0)
d_dut_dc.plot_z_time_step(0, 0)
plt.xlim(-2, 4)

# residuals frequency domain
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
res.plot_s_db(1, 0)
plt.subplot(1, 2, 2)
res.plot_s_deg(1, 0)
ssaver('./test/test_rf/test_data/test5.png')

start_time = time.time()
print(f"Time cost: {start_time - end_time} seconds")
