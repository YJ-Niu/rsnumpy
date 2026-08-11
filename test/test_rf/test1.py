import warnings
import rsplotlib.pyplot as plt
import rsnumpy as np
from rsnumpy import absolute, log10, real, sum
from scipy.optimize import minimize
from skrf.calibration.deembedding import IEEEP370_SE_NZC_2xThru
from skrf.media import CPW
import time
import skrf as rf
from skrf.media import MLine

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
ssaver('./test/test_rf/test_data/test5.png')
# residuals frequency domain
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
res.plot_s_db(1, 0)
plt.subplot(1, 2, 2)
res.plot_s_deg(1, 0)
ssaver('./test/test_rf/test_data/test6.png')


plt.figure()
c0 = 3e8
f = MSL100.f
deltaL = 0.1
deltaPhi = np.unwrap(np.angle(MSL100.s[:, 1, 0])) - np.unwrap(np.angle(MSL200.s[:, 1, 0]))
Er_eff = np.power(deltaPhi * c0 / (2 * np.pi * f * deltaL), 2)
Loss_mea = 20 * log10(absolute(MSL200.s[:, 1, 0] / MSL100.s[:, 1, 0]))

W = 3.00e-3
H = 1.55e-3
T = 50e-6
L = 0.1
Er0 = 4.5
tand0 = 0.02
f_epr_tand = 1e9
x0 = [Er0, tand0]
def model(x, freq, Er_eff, L, W, H, T, f_epr_tand, Loss_mea):
    ep_r = x[0]
    tand = x[1]
    m = MLine(frequency=freq, z0_port=50, w=W, h=H, t=T,
              ep_r=ep_r, mu_r=1, rho=1.712e-8, tand=tand, rough=0.15e-6,
              f_low=1e3, f_high=1e12, f_epr_tand=f_epr_tand,
              diel='djordjevicsvensson', disp='kirschningjansen')
    DUT = m.line(L, 'm')
    Loss_mod = 20 * log10(absolute(DUT.s[:, 1, 0]))
    return sum((real(m.ep_reff_f) - Er_eff)**2) + 0.01*sum((Loss_mod - Loss_mea)**2)


res = minimize(model, x0, args=(MSL100.frequency, Er_eff, L, W, H, T, f_epr_tand, Loss_mea),
               bounds=[(4.2, 4.7), (0.001, 0.1)])
Er = res.x[0]
tand = res.x[1]

m = MLine(frequency=MSL100.frequency, z0_port=50, w=W, h=H, t=T,
          ep_r=Er, mu_r=1, rho=1.712e-8, tand=tand, rough=0.15e-6,
          f_low=1e3, f_high=1e12, f_epr_tand=f_epr_tand,
          diel='djordjevicsvensson', disp='kirschningjansen')
DUT = m.line(L, 'm')
DUT.name = 'DUT'
plt.title('Measured vs modelled data')
MSL100.plot_s_db()
DUT.plot_s_db(0, 0, color='k')
DUT.plot_s_db(1, 0, color='k')
ssaver('./test/test_rf/test_data/test7.png')

phi_conn = np.unwrap(np.angle(MSL100.s[:, 1, 0])) + deltaPhi
z = np.polyfit(f, phi_conn, 1)
p = np.poly1d(z)
delay = -z[0]/(2*np.pi)/2
print(f'Connector delay: {delay * 1e12:.0f} ps')

loss_conn_db = 20 * log10(absolute(MSL100.s[:, 1, 0])) - Loss_mea
alpha = 1.6*np.log(10)/20 * np.sqrt(f/1e9)
beta = 2*np.pi*f/c0
gamma = alpha + 1j*beta
mf = rf.media.DefinedGammaZ0(m.frequency, z0_port=50, z0=55.0, gamma=gamma)
left = mf.line(delay*1e9, 'ns')
right = left.flipped()
check = left ** right

plt.figure()
plt.suptitle('Connector effects')
plt.subplot(2, 1, 1)
plt.plot(f * 1e-9, phi_conn, label='measured')
plt.plot(f * 1e-9, np.unwrap(np.angle(check.s[:, 1, 0])), label='model')
plt.ylabel('phase (rad)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(f * 1e-9, loss_conn_db, label='Measured')
plt.plot(f * 1e-9, 20*np.log10(np.absolute(check.s[:, 1, 0])), label='model')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Insertion Loss (dB)')
plt.legend()
ssaver('./test/test_rf/test_data/test8.png')

mod = left ** DUT ** right
mod.name = 'Model'

MSL100_dc = MSL100.extrapolate_to_dc(kind='cubic')
DUT_dc = mod.extrapolate_to_dc(kind='cubic')

plt.figure()
plt.suptitle('Left-right and right-left TDR')
plt.subplot(2, 1, 1)
MSL100_dc.plot_z_time_step(0, 0)
DUT_dc.plot_z_time_step(0, 0)
plt.xlim(-2, 4)

plt.subplot(2, 1, 2)
MSL100_dc.plot_z_time_step(1, 1)
DUT_dc.plot_z_time_step(1, 1)
plt.xlim(-2, 4)
plt.tight_layout()
ssaver('./test/test_rf/test_data/test9.png')

plt.figure()
plt.title('Measured vs modelled data')
MSL100.plot_s_db()
mod.name = 'Model'
mod.plot_s_db(0, 0, color='k')
mod.plot_s_db(1, 0, color='k')
ssaver('./test/test_rf/test_data/test10.png')
start_time = time.time()
print(f"Time cost: {start_time - end_time} seconds")
