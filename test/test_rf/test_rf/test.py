import skrf as rf
from skrf import Frequency, Network
from skrf.data import ring_slot  # noqa: F811
import rsnumpy as np
import rsplotlib.pyplot as plt
from rsplotlib import style
from skrf.networkSet import NetworkSet
from skrf.media import CPW, Coaxial
from skrf.data import wr2p2_line1 as line1
from skrf.data import wr1p5_line, wr2p2_line
import os
from skrf.data import ring_slot_meas
from skrf.plotting import save_all_figs
from skrf import plotting
from skrf.circuit import Circuit
from skrf.calibration.deembedding import OpenShort
from skrf.constants import to_meters
from skrf.media import MLine, RectangularWaveguide
from skrf.media import Freespace
from scipy.optimize import fmin
from skrf.network import concat_ports, connect
from skrf.constants import c

def pprint(n, ss):
    print(f"Network {n}")
    print("++++++++++++++++++++++++++++++")
    print(ss, "\n")


# 保存当前图
def ssaver(name):
    plt.savefig(name)
    plt.clf()


# ring_slot = rf.Network('data/ring slot.s2p')

# ring_slot
pprint(1, ring_slot)
short = rf.data.wr2p2_short
delayshort = rf.data.wr2p2_delayshort

pprint(2, short - delayshort)
pprint(3, short/delayshort)

short = rf.data.wr2p2_short
line = rf.data.wr2p2_line

delayshort = line ** short
short = line.inv ** delayshort
pprint(4, short)
print(type(line.s))
print(line.s.shape)

print(line.frequency)
pprint(5, line.f[0:10])

rs = rf.data.ring_slot  # another 2-port example
pprint(6, rs.s_mag[:, 1, 0].min())
f_match = rs.f[np.argmin(rs.s_mag[:, 0, 0])]  # frequency for min(|S11|)
pprint(7, f_match)

# rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_db()
# ssaver('./test/test_rf/test1.png')


# rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_deg(m=0, n=1)
# ssaver('./test/test_rf/test2.png')


# rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_smith(lw=2)
plt.title('Big ole Smith Chart')
# ssaver('./test/test_rf/test3.png')


print(rf.io.read_all('./test/test_rf/skrf/data/', contains='ro'))
ro_dict = rf.io.read_all('./test/test_rf/skrf/data/', contains='ro')
ro_ns = NetworkSet(ro_dict, name='ro set')  # name is optional
print(ro_ns)
pprint(8, ro_ns.mean_s)
pprint(9, ro_ns.std_s)

ro_ns.std_s.plot_s_mag(label='S11')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of RO')
plt.legend()
# ssaver('./test/test_rf/ro_std_s.png')


ro_ns.plot_uncertainty_bounds_s_db(label='S11')
# ssaver('./test/test_rf/test4.png')


freq = Frequency(75, 110, 101, 'GHz')
cpw = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)
pprint(10, cpw)

pprint(11, cpw.line(d=90, unit='deg', name='line'))

freq = Frequency(1, 10, 101, 'GHz')
coax = Coaxial(frequency=freq, Dint=1e-3, Dout=2e-3)
pprint(12, coax)

# dummy 2-port network from Frequency and s-parameters
freq = Frequency(1, 10, 101, 'ghz')
rng = np.random.default_rng()
s = rng.uniform(size=(101, 2, 2)) + 1j*rng.uniform(size=(101, 2, 2))  # random complex numbers
# if not passed, will assume z0=50. name is optional but it's a good practice.
ntwk = Network(frequency=freq, s=s, name='random values 2-port')
pprint(13, ntwk)

ntwk.plot_s_db()
# ssaver('./test/test_rf/test5.png')


# let's assume we have separate arrays for the frequency and s-parameters
f = np.array([1, 2, 3, 4])  # in GHz
S11 = rng.uniform(size=4)
S12 = rng.uniform(size=4)
S21 = rng.uniform(size=4)
S22 = rng.uniform(size=4)

# Before creating the scikit-rf Network object, one must forge the Frequency and S-matrix:
freq2 = rf.Frequency.from_f(f, unit='GHz')

# forging S-matrix as shape (nb_f, 2, 2)
# there is probably smarter way, but less explicit for the purpose of this example:
s = np.zeros((len(f), 2, 2), dtype=complex)
s[:, 0, 0] = S11
s[:, 0, 1] = S12
s[:, 1, 0] = S21
s[:, 1, 1] = S22

# constructing Network object
ntw = rf.Network(frequency=freq2, s=s)

pprint(14, ntw)

ntw2 = rf.Network(frequency=freq, s=s, z0=25, name='same z0 for all ports')
pprint(15, ntw2)
ntw3 = rf.Network(frequency=freq, s=s, z0=[20, 30], name='different z0 for each port')
pprint(16, ntw3)
ntw4 = rf.Network(frequency=freq, s=s, z0=rng.uniform(size=(4, 2)), name='different z0 for each frequencies and ports')
pprint(17, ntw4)

# 1-port network example
z = np.full((len(freq), 1, 1), 10j)  # replicate z=10j for all frequencies

ntw = rf.Network(frequency=freq, z=z)
pprint(18, ntw)

z = 20
abcd = np.array([[1, z], [0, 1]])

a = np.tile(abcd, (len(freq), 1, 1))
ntw = Network(frequency=freq, a=a)
pprint(19, ntw)

s = rf.network.a2s(a)
# checking that these S-params are the same
pprint(20, np.all(ntw.s == s))

pprint(21, np.shape(ring_slot.s))


s_a = ring_slot.s[:11, 1, 0]  # get first 10 values of S21
pprint(22, s_a)

pprint(23, ring_slot[0:10])

pprint(24, ring_slot['80-90ghz'])

pprint(25, ring_slot.s11['80-90ghz'])

rf.stylely()
ring_slot.plot_s_smith()
# ssaver('./test/test_rf/test6.png')


plt.title('Ring Slot $S_{21}$')

rf.stylely()
ring_slot.s11.plot_s_db(label='Full Band Response')
ring_slot.s11['82-90ghz'].plot_s_db(lw=3, label='Band of Interest')
plt.legend()
# ssaver('./test/test_rf/test7.png')

short - delayshort
short + delayshort
short * delayshort
pprint(26, short / delayshort)

difference = (short - delayshort)
difference.plot_s_mag(label='Mag of difference')
# ssaver('./test/test_rf/test8.png')


(delayshort/short).plot_s_deg(label='Detrended Phase')
# ssaver('./test/test_rf/test9.png')


hopen = (short*-1)
pprint(27, hopen.s[:3, ...])

rando = hopen * rng.uniform(size=len(hopen))
pprint(28, rando.s[:3, ...])

pprint(29, short == delayshort)
pprint(30, short != delayshort)

short = rf.data.wr2p2_short
line = rf.data.wr2p2_line
delayshort = line ** short
short_2 = line.inv ** delayshort

pprint(31, short_2 == short)

tee = rf.data.tee
pprint(32, tee)

terminated_tee = rf.network.connect(tee, 1, delayshort, 0)
pprint(33, terminated_tee)

pprint(34, line)
pprint(35, line1)
line1.resample(201)
pprint(36, line1)

line1 + line
print("---------------------")
# 11111111111111
big_line = rf.network.stitch(wr2p2_line, wr1p5_line)
pprint(37, big_line)
pprint(38, wr2p2_line)
pprint(39, wr1p5_line)

rf.io.write('./test/test_rf/skrf/data/myline.ntwk', line)

ntwk = Network('./test/test_rf/skrf/data/myline.ntwk')  # read Network using pickle
dict_o_ntwks = rf.io.read_all(rf.data.pwd, contains='wr2p2')
pprint(40, dict_o_ntwks)


dict_o_ntwks_files = rf.io.read_all(
    files=[os.path.join(rf.data.pwd, test_file) for test_file in ['ntwk1.s2p', 'ntwk2.s2p']]
)
pprint(41, dict_o_ntwks_files)

pprint(42, ring_slot.z[:3, ...])

ring_slot.plot_z_im(m=1, n=0)
# ssaver('./test/test_rf/test10.png')


line = rf.data.wr2p2_line  # 2-port
short = rf.data.wr2p2_short  # 1-port

delayshort = line ** short  # --> 1-port Network
pprint(43, delayshort)

delayshort2 = rf.network.cascade(line, short)
pprint(44, delayshort2 == delayshort)  # the result is the same

delayshort3 = rf.network.connect(line, 1, short, 0)
pprint(45, delayshort3 == delayshort)  # the result is the same

line1 = rf.data.wr2p2_line  # 2-port
line2 = rf.data.wr2p2_line  # 2-port
line3 = rf.data.wr2p2_line  # 2-port
line4 = rf.data.wr2p2_line  # 2-port
short = rf.data.wr2p2_short  # 1-port

chain1 = line1 ** line2 ** line3 ** line4 ** short

chain2 = rf.network.cascade_list([line1, line2, line3, line4, short])

pprint(46, chain1 == chain2)

tee = rf.data.tee
terminated_tee = rf.network.connect(tee, 1, delayshort, 0)
pprint(47, terminated_tee)

terminated_tee_par = rf.network.parallelconnect([tee, delayshort], [1, 0])
pprint(48, terminated_tee_par)

tee.z0 = [1, 2, 3]
line.z0 = [10, 20]
# the resulting network is:
pprint(49, rf.network.connect(tee, 1, line, 0))

ring_slot.plot_s_smith()
# ssaver("./test/test_rf/test11.png")


rf.stylely()  # nicer looking. Can be configured with different styles
ring_slot.plot_s_smith()
# ssaver("./test/test_rf/test12.png")


ring_slot.plot_s_smith(draw_labels=True)
# ssaver("./test/test_rf/test13.png")


ring_slot.plot_s_smith(chart_type='y')
# ssaver("./test/test_rf/test14.png")


# prepare markers
lines = [
    {'marker_idx': [30, 60, 90], 'color': 'g', 'm': 0, 'n': 0, 'ntw': ring_slot},
    {'marker_idx': [15, 45, 75], 'color': 'r', 'm': 1, 'n': 0, 'ntw': ring_slot},
]

# prepare figure
fig, ax = plt.subplots(1, 1, figsize=(7, 8))

# impedance smith chart
rf.plotting.smith(ax=ax, draw_labels=True, ref_imm=50.0, chart_type='z')

# plot data
col_labels = ['Frequency', 'Real Imag']
row_labels = []
row_colors = []
cell_text = []
for l_ in lines:
    m = l_['m']
    n = l_['n']
    l_['ntw'].plot_s_smith(m=m, n=n, ax=ax, color=l_['color'], draw_chart=False)
    # plot markers
    for i, k in enumerate(l_['marker_idx']):
        x = l_['ntw'].s.real[k, m, n]
        y = l_['ntw'].s.imag[k, m, n]
        z = l_['ntw'].z[k, m, n]
        z = f'{z.real:.4f} + {z.imag:.4f}j ohm'
        f = l_['ntw'].frequency.f_scaled[k]
        f_unit = l_['ntw'].frequency.unit
        row_labels.append(f'M{i + 1}')
        row_colors.append(l_['color'])
        ax.scatter(x, y, marker='v', s=20, color=l_['color'])
        ax.annotate(row_labels[-1], (x, y), xytext=(-7, 7), textcoords='offset points', color=l_['color'])
        cell_text.append([f'{f:.3f} {f_unit}', z])

ax.legend(fontsize=6, loc='upper right')

# plot the table
the_table = ax.table(cellText=cell_text,
                     colWidths=[0.4] * 2,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     rowColours=row_colors,
                     loc='bottom')
the_table.auto_set_font_size(False)
the_table.set_fontsize(6)
# the_table.scale(1.5, 1.5)
# ssaver("./test/test_rf/test15.png")


# prepare figure
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# background = plt.imread('figures/smithchart.png')

# tweak background position
# ax.imshow(background, extent=[-1.185, 1.14, -1.13, 1.155])
rf.plotting.smith(ax=ax, draw_labels=True, ref_imm=1.0, chart_type='z')

ring_slot.plot_s_smith(ax=ax, draw_chart=False)
# ssaver("./test/test_rf/test16.png")


ring_slot.plot_s_complex()

rf.stylely()
plt.axis('equal')  # otherwise circles won't be circles
# ssaver("./test/test_rf/test17.png")


rf.stylely()
ring_slot.plot_s_db()
# ssaver("./test/test_rf/test18.png")


rf.stylely()
ring_slot.plot_s_db(m=0, n=0, label='Theory')
ring_slot_meas.plot_s_db(m=0, n=0, label='Measurement')
# ssaver("./test/test_rf/test19.png")


ring_slot.plot_s_deg()
# ssaver("./test/test_rf/test20.png")


ring_slot.plot_s_deg_unwrap()
# ssaver("./test/test_rf/test21.png")


gd = abs(ring_slot.s21.group_delay) * 1e9  # in ns

ring_slot.plot(gd)
plt.ylabel('Group Delay (ns)')
plt.title('Group Delay of Ring Slot S21')
# ssaver("./test/test_rf/test22.png")


ring_slot.plot_z_im()
# ssaver('./test/test_rf/test23.png')


ring_slot.plot_y_im()
# ssaver('./test/test_rf/test24.png')


ring_slot.plot_s_db(m=0, n=0, label='Simulation')
# ssaver('./test/test_rf/test25.png')


ring_slot.frequency.unit = 'mhz'
ring_slot.plot_s_db(0, 0)
# ssaver('./test/test_rf/test26.png')


ring_slot.frequency.unit = 'ghz'
ring_slot.plot_s_db(m=0, n=0, linewidth=3, linestyle='--', label='Simulation')
ring_slot_meas.plot_s_db(m=0, n=0, marker='<', markevery=10, label='Measured')
# ssaver('./test/test_rf/test27.png')


mpl_style = "seaborn-ticks"
try:
    mpl_style = mpl_style if mpl_style in style.available else "seaborn-v0_8-ticks"
except Exception:
    mpl_style = "seaborn-v0_8-ticks"
with style.context(mpl_style):
    ring_slot.plot_s_smith()
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Smith Chart With Legend Room')
    plt.axis([-1.1, 2.1, -1.1, 1.1])
    plt.legend(loc=5)
# ssaver('./test/test_rf/test28.png')


save_all_figs('./test/test_rf', format=['png'])


with plt.style.context('grayscale'):
    ring_slot.plot_s_deg()
    plotting.add_markers_to_lines()
    plt.legend()  # have to re-generate legend

    # ssaver('./test/test_rf/test29.png')

pprint(50, rf.io.read_all(rf.data.pwd, contains='ro'))

ro_dict = rf.io.read_all(rf.data.pwd, contains='ro')
ro_ns = NetworkSet(ro_dict, name='ro set')
pprint(51, ro_ns)

pprint(52, ro_ns[0])

rf.stylely()
ro_ns.plot_s_db()
# ssaver('./test/test_rf/test30.png')


pprint(53, ro_ns.mean_s)
ro_ns.mean_s.plot_s_db(label='ro')
# ssaver('./test/test_rf/test31.png')


ro_ns.std_s.plot_s_re(y_label='Standard Deviations')
# ssaver('./test/test_rf/test32.png')


ro_ns.mean_s_deg.plot_s_re()
# ssaver('./test/test_rf/test33.png')

plt.close()

ro_ns.plot_uncertainty_bounds_s_db()
# ssaver('./test/test_rf/test34.png')


ro_ns.plot_uncertainty_bounds_s_deg()
# ssaver('./test/test_rf/test35.png')


rf.stylely()
ro_ns_interp = ro_ns.interpolate_frequency(rf.Frequency(500, 600, 15, "GHz"))
ro_ns_interp.plot_violin("s_db")
# ssaver('./test/test_rf/test36.png')


rf.stylely()
ro_ns_interp.plot_violin("s_deg")
# ssaver('./test/test_rf/test37.png')


pprint(58, ro_ns.write_touchstone(dir='./test/test_rf/test_data'))

rf.io.write('./test/test_rf/test_data/ro set.ns', ro_ns)

ro_ns = rf.io.read('./test/test_rf/test_data/ro set.ns')
pprint(59, ro_ns)

ro_ns.write_spreadsheet('./test/test_rf/test_data/ro_spreadsheet.csv', form='db')

params = [{'a': 0, 'X': 10, 'c': 'A'},
          {'a': 1, 'X': 10, 'c': 'A'},
          {'a': 2, 'X': 10, 'c': 'A'},
          {'a': 1, 'X': 20, 'c': 'A'},
          {'a': 0, 'X': 20, 'c': 'A'},
          ]
# create a NetworkSet made of dummy Networks, each define for set of parameters
freq1 = rf.Frequency(75, 110, 101, 'ghz')
rng = np.random.default_rng()
ntwks_params = [rf.Network(frequency=freq1, s=rng.uniform(size=(len(freq1), 2, 2)),
                           name=f'ntwk_{m}', comment=f'ntwk_{m}', params=params) for (m, params) in enumerate(params)]
ns = rf.networkSet.NetworkSet(ntwks_params)
pprint(60, ns)

pprint(61, ns.sel({'a': 1}))

pprint(62, ns.sel({'a': 0, 'X': 10}))

pprint(63, ns.sel({'a': 0, 'X': [10, 20]}))

pprint(64, ns.sel({'a': [0, 1], 'X': [10, 20]}))
pprint(65, ns.dims)

pprint(66, ns.coords)

param_x = [1, 2, 3]  # a parameter associated to each Network
x0 = 1.5  # parameter value to interpolate for
interp_ntwk = ro_ns.interpolate_from_network(param_x, x0)
pprint(67, interp_ntwk)

rf.stylely()
Z_0 = 50
Z_L = 75
theta = 0

# the necessary Frequency description
freq = rf.Frequency(start=1, stop=2, unit='GHz', npoints=3)

# The combination of a transmission line + a load can be created
# using the convenience delay_load method
# important: all the Network must have the parameter "name" defined
tline_media = rf.media.DefinedGammaZ0(freq, z0=Z_0)
delay_load = tline_media.delay_load(rf.tlineFunctions.zl_2_Gamma0(Z_0, Z_L), theta, unit='deg', name='delay_load')

# the input port of the circuit is defined with the Circuit.Port method
port1 = Circuit.Port(freq, 'port1', z0=Z_0)

# connection list
cnx = [
    [(port1, 0), (delay_load, 0)]
]
# building the circuit
cir = Circuit(cnx)

# getting the resulting Network from the 'network' parameter:
ntw = cir.network
pprint(68, ntw)

# as expected the reflection coefficient is:
pprint(69, ntw.s[0])

port1 = Circuit.Port(freq, 'port1', z0=Z_0)
# piece of transmission line and series impedance
trans_line = tline_media.line(theta, unit='deg', name='trans_line')
load = tline_media.resistor(Z_L, name='delay_load')
# ground network (short)
ground = Circuit.Ground(freq, name='ground')

# connection list
cnx = [
    [(port1, 0), (trans_line, 0)],
    [(trans_line, 1), (load, 0)],
    [(load, 1), (ground, 0)]
]
# building the circuit
cir = Circuit(cnx)
# the result if the same :
pprint(70, cir.network.s[0])

fig.clear()
fig = plt.figure(figsize=(10, 6))
freq = rf.Frequency(start=0.1, stop=10, unit='GHz', npoints=1001)
tl_media = rf.media.DefinedGammaZ0(freq, z0=50, gamma=1j*freq.w/rf.constants.c)
C1 = tl_media.capacitor(3.222e-12, name='C1')
C2 = tl_media.capacitor(82.25e-15, name='C2')
C3 = tl_media.capacitor(3.222e-12, name='C3')
L2 = tl_media.inductor(8.893e-9, name='L2')
RL = tl_media.resistor(50, name='RL')
gnd = Circuit.Ground(freq, name='gnd')
port1 = Circuit.Port(freq, name='port1', z0=50)
port2 = Circuit.Port(freq, name='port2', z0=50)

cnx = [
    [(port1, 0), (C1, 0), (L2, 0), (C2, 0)],
    [(L2, 1), (C2, 1), (C3, 0), (port2, 0)],
    [(gnd, 0), (C1, 1), (C3, 1)],
]
cir = Circuit(cnx)
ntw = cir.network
ntw.plot_s_db(m=0, n=0, lw=2, logx=True)
ntw.plot_s_db(m=1, n=0, lw=2, logx=True)
ssaver('./test/test_rf/test38.png')

cir.plot_graph(network_labels=True, network_fontsize=15,
               port_labels=True, port_fontsize=15,
               edge_labels=True, edge_fontsize=10)

ssaver('./test/test_rf/test39.png')

freq = rf.Frequency(50, 200, 301, 'mhz')
tl_media = rf.media.DefinedGammaZ0(frequency=freq, z0=50)
gnd = Circuit.Ground(frequency=freq, name='ground')
C1 = tl_media.capacitor(6.353e-12, name='C1')
L1 = tl_media.inductor(402.7e-9, name='L1')
C2 = tl_media.capacitor(61.08e-12, name='C2')
L2 = tl_media.inductor(13.63e-9, name='L2')
C3 = tl_media.capacitor(187.8e-12, name='C3')
L3 = tl_media.inductor(41.89e-9, name='L3')
C4 = tl_media.capacitor(6.353e-12, name='C4')
L4 = tl_media.inductor(402.27e-9, name='L4')
Port1 = Circuit.Port(frequency=freq, name='PortIn')
Port2 = Circuit.Port(frequency=freq, name='PortOut')
cnx = [
    [(Port1, 0), (C1, 0)],
    [(C1, 1), (L1, 0)],
    [(L1, 1), (C2, 0), (C3, 0), (C4, 0)],
    [(C2, 1), (L2, 0)],
    [(C3, 1), (L3, 0)],
    [(L2, 1), (L3, 1), (gnd, 0)],
    [(C4, 1), (L4, 0)],
    [(L4, 1), (Port2, 0)]
]
cir = Circuit(cnx)
ntw = cir.network

ntw.plot_s_db(m=0, n=0, lw=2, logx=True)
ntw.plot_s_db(m=1, n=0, lw=2, logx=True)
ssaver('./test/test_rf/test40.png')

# Look at the raw inductor measurement with parasitics included
# From S11/S22, it is clear that it is not a pure inductance.
raw_ind = rf.data.ind
raw_ind.plot_s_smith()
ssaver('./test/test_rf/test41.png')

Lraw_nH = 1e9 * np.imag(1/raw_ind.y[:, 0, 0])/2/np.pi/raw_ind.f
Qraw = np.abs(np.imag(1/raw_ind.y[:, 0, 0])/np.real(1/raw_ind.y[:, 0, 0]))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(raw_ind.f*1e-9, Lraw_nH)
ax1.grid()
ax1.set_ylabel("Inductance (nH)")
ax1.set_xlabel("Freq. (GHz)")
ax2.plot(raw_ind.f*1e-9, Qraw)
ax2.grid()
ax2.set_ylabel("Q-factor")
ax2.set_xlabel("Freq. (GHz)")
fig.tight_layout()
ssaver('./test/test_rf/test42.png')
fig.clear()

fig = plt.figure(figsize=(10, 10))
# load in 2-ports short/open dummy networks
open_nw = rf.data.open_2p
short_nw = rf.data.short_2p

dm = OpenShort(dummy_open=open_nw, dummy_short=short_nw, name='tutorial')

actual_ind = dm.deembed(raw_ind)
actual_ind.plot_s_smith()
ssaver('./test/test_rf/test43.png')

rf.stylely()
C = 1e-6  # F
L = 1e-9  # H
R = 30  # Ohm
Z0 = 50  # Ohm

freq = rf.Frequency(5, 5.2, npoints=501, unit='MHz')
media = rf.media.DefinedGammaZ0(frequency=freq, z0=Z0)  # ideal line (no loss)
rng = np.random.default_rng()
random_d = rng.uniform(-np.pi, np.pi)  # a random length for the sake of the example

resonator = media.line(d=random_d, unit='rad') ** media.shunt_inductor(L) ** media.shunt_capacitor(C) ** media.shunt(media.resistor(R)**media.short()) ** media.open()

resonator.plot_s_db()
ssaver('./test/test_rf/test44.png')

Lactual_nH = 1e9 * np.imag(1/actual_ind.y[:, 0, 0])/2/np.pi/actual_ind.f

fig, ax1 = plt.subplots(1, 1)
ax1.plot(actual_ind.f*1e-9, Lactual_nH)
ax1.grid()
ax1.set_ylim(0.95, 1.1)
ax1.set_ylabel("Inductance (nH)")
ax1.set_xlabel("Freq. (GHz)")
fig.tight_layout()
ssaver('./test/test_rf/test45.png')

C = 1e-6  # F
L = 1e-9  # H
R = 30  # Ohm
Z0 = 50  # Ohm

freq = rf.Frequency(5, 5.2, npoints=501, unit='MHz')
media = rf.media.DefinedGammaZ0(frequency=freq, z0=Z0)  # ideal line (no loss)
rng = np.random.default_rng()
random_d = rng.uniform(-np.pi, np.pi)  # a random length for the sake of the example

resonator = media.line(d=random_d, unit='rad') ** media.shunt_inductor(L) ** media.shunt_capacitor(C) ** media.shunt(media.resistor(R)**media.short()) ** media.open()

resonator.plot_s_db()
ssaver('./test/test_rf/test46.png')

def f_res_RLC(L, C):
    return 1/(2*np.pi*np.sqrt(L*C))

def Q_RLC(R, L, C):
    return R * C / np.sqrt(L*C)


pprint(71, f'Theoretical Resonant Frequency: {f_res_RLC(L, C)/1e6} MHz')
pprint(72, f'Theoretical Loaded Q: Q_L = {Q_RLC((R*Z0)/(R+Z0), L, C)}')  # Req = R//Z0
pprint(73, f'Theoretical Unloaded Q: Q_0 = {Q_RLC(R, L, C)}')

Q = rf.qfactor.Qfactor(resonator, res_type='reflection')

res = Q.fit()
pprint(74, f'Fitted Resonant Frequency: f_L = {Q.f_L/1e6} MHz')
pprint(75, f'Fitted Loaded Q-factor: Q_L = {Q.Q_L}')

pprint(76, Q)

Q0 = Q.Q_unloaded(res)
pprint(77, f'Fitted Unloaded Q-factor: Q_0 = {Q0}')

Q0 = Q.Q_unloaded()  # will use the latest optimized results performed with .fit()
pprint(78, f'Fitted Unloaded Q-factor: Q_0 = {Q0}')
pprint(79, f'Relative Error on Q_0: {(Q_RLC(R, L, C) - Q0)/Q_RLC(R, L, C)}')

new_freq = rf.Frequency(5, 5.2, npoints=5001, unit='MHz')
fitted_network = Q.fitted_network(res, frequency=new_freq)
resonator.plot_s_mag(label='Parallel RLC ', lw=2)
fitted_network.plot_s_mag(label='Fitted Model', lw=2, ls='--')
ssaver('./test/test_rf/test47.png')
fig.clear()
plt.clf()
plt.close()

diam, S_V, S_T = Q.Q_circle()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
resonator.plot_s_polar(ax=ax, label="RLC Resonator", ls='', marker='x', ms=5)
fitted_network.plot_s_polar(ax=ax, label="Fitted Model", lw=2)
ax.plot(np.angle(S_V), np.abs(S_V), 'ko')
ax.plot(np.angle(S_T), np.abs(S_T), 'ko')
ax.text(np.angle(S_T), 0.8*np.abs(S_T), '$S_T$')
ax.text(np.angle(S_V), 1.1*np.abs(S_V), '$S_V$')

ssaver('./test/test_rf/test48.png')

print(111, np.angle(S_T), 0.8*np.abs(S_T), S_T)
print(222, np.angle(S_V), 1.1*np.abs(S_V), S_V)


BW = Q.BW
pprint(80, f'Bandwidth: {BW} Hz')
fig, ax = plt.subplots()
rf.stylely()

resonator.plot_s_db(label='Parallel RLC ', lw=2, ax=ax)
ax.axvspan(xmin=(Q.f_L-Q.BW/2)/1e6, xmax=(Q.f_L+Q.BW/2)/1e6, alpha=0.3, label='Bandwidth')
ax.legend()
ssaver('./test/test_rf/test49.png')

# frequency
f_rg58 = Frequency(1, 5, 101, 'GHz')

# media with z0_port the port impedance of the VNA
rg58 = Coaxial(f_rg58, Dint=0.91e-3, Dout=2.95e-3, epsilon_r=2.3, z0_port=50)
pprint(81, rg58)

print(f'z0 = {rg58.z0[0]}')
print(f'z0_port = {rg58.z0_port[0]}')
print(f'gamma = {rg58.gamma[0]}')

rg58_line = rg58.line(100, unit='mm', name='100 mm, z0 Ohm')
pprint(85, rg58_line)

rg58_25ohm_line = rg58.line(100, unit='mm', z0=25, name='100 mm, 25 Ohm')
pprint(86, rg58_25ohm_line)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
# return loss
rg58_line.plot_s_db(0, 0, ax=axes[0])
rg58_25ohm_line.plot_s_db(0, 0, ax=axes[0])
axes[0].set_title('Return Loss')
rg58_line.plot_s_db(1, 0, ax=axes[1], label='100 mm, z0 Ohm S11')
rg58_25ohm_line.plot_s_db(1, 0, ax=axes[1], label='100 mm, 25 Ohm S11')
axes[1].set_title('Insertion Loss')
plt.tight_layout()
rf.stylely()
ssaver('./test/test_rf/test50.png')

rf.stylely()

# create frequency axes
f_mlin = Frequency(0.1, 10, 1001, 'GHz')
f_wr10 = Frequency(75, 110, 1001, 'GHz')

# create media from parameters
mlin = MLine(f_mlin, w=3e-3, h=1.6e-3, t=35e-6, ep_r=4.5, rho=1.68e-08)
print(mlin)
wr10 = RectangularWaveguide(f_wr10, a=to_meters(100, 'mil'), b=to_meters(50, 'mil'), ep_r=1.0, rho=1.68e-08)
print(wr10)

# create the transmission line networks
mlin_100 = mlin.line(100e-3, unit='m', name='mlin 100mm')
print(mlin_100)
wr10_100 = wr10.line(100e-3, unit='m', name='wr10 100mm')
print(wr10_100)

# prepare figure
fig1, axes = plt.subplots(2, 2, figsize=(8, 6))
rf.stylely()

# plot miscrostipline
mlin_100.plot_s_mag(0, 0, ax=axes[0, 0])
mlin_100.plot_s_db(1, 0, ax=axes[0, 1])

# plot rectangular waveguide
wr10_100.plot_s_mag(0, 0, ax=axes[1, 0])
wr10_100.plot_s_db(1, 0, ax=axes[1, 1])

# resize plot nicely
axes[0, 0].set_ylim((-1, 1))
axes[1, 0].set_ylim((-1, 1))
fig1.tight_layout()
ssaver('./test/test_rf/test51.png')

fig1.clear()
plt.close()

fig2, axes = plt.subplots(1, 2, figsize=(10, 3.5))
# plot miscrostipline
rf.stylely()
axes[0].plot(mlin_100.frequency.f_scaled, mlin_100.z0[:, 0].real, marker='.', label=f'line {mlin_100.name}  port z0')
axes[0].plot(mlin.frequency.f_scaled, mlin.z0.real, label='media mlin characteristic z0')
axes[0].set_ylabel('Impedance (Ohm)')
axes[0].set_xlabel(f'Frequency ({mlin.frequency.unit})')
axes[0].set_title('Microstripline')
axes[0].legend()

# plot rectangular waveguide
axes[1].plot(wr10_100.frequency.f_scaled, wr10_100.z0[:, 0].real, marker='.', label=f'line {wr10_100.name} port z0')
axes[1].plot(wr10.frequency.f_scaled, wr10.z0.real, label='media wr10 characteristic z0')
axes[1].set_ylabel('Impedance (Ohm)')
axes[1].set_xlabel(f'Frequency ({wr10.frequency.unit})')
axes[1].set_title('WR-10')
axes[1].legend()

# resize plot nicely
fig2.tight_layout()
ssaver('./test/test_rf/test52.png')

# renormalization method
mlin_100_measured1 = mlin_100.copy()
mlin_100_measured1.renormalize([50, 50])
mlin_100_measured1.name = f'{mlin_100.name} renormalize'
print(mlin_100_measured1)

# port impedance specified at media construction method
mlin_meas = MLine(f_mlin, w=3e-3, h=1.6e-3, t=35e-6, ep_r=4.5, rho=1.68e-08, z0_port=50)
mlin_100_measured2 = mlin_meas.line(100e-3, unit='m', name='mlin 100mm z0_port')
pprint(87, mlin_100_measured2)


# prepare figure
fig3, axes = plt.subplots(2, 2, figsize=(10, 6))
rf.stylely()
gs = axes[1, 0].get_gridspec()
for ax in axes[1, :]:
    ax.remove()
axbig = fig3.add_subplot(gs[1, :])

# plot return loss
mlin_100_measured1.plot_s_db(0, 0, ax=axes[0, 0])
mlin_100_measured2.plot_s_db(0, 0, ax=axes[0, 0])

# plot insertion loss
mlin_100_measured1.plot_s_db(1, 0, ax=axes[0, 1])
mlin_100_measured2.plot_s_db(1, 0, ax=axes[0, 1])
# plot port and characteristic impedances
axbig.plot(mlin_100_measured1.frequency.f_scaled, mlin_100_measured1.z0[:, 0].real, marker='d', markevery=30, label=f'line {mlin_100_measured1.name} z0')
axbig.plot(mlin_100_measured2.frequency.f_scaled, mlin_100_measured2.z0[:, 0].real, marker='x', markevery=30, label=f'line {mlin_100_measured2.name} z0')
axbig.plot(mlin.frequency.f_scaled, mlin.z0.real, label='media mlin z0')
axbig.set_ylabel('Impedance (Ohm)')
axbig.set_xlabel(f'Frequency ({mlin.frequency.unit})')
plt.legend(fontsize=6)

# resize plot nicely
fig3.tight_layout()
ssaver('./test/test_rf/test53.png')

# override method
wr10_100_measured1 = wr10_100.copy()
wr10_100_measured1.z0[:, :] = 50
wr10_100_measured1.name = f'{wr10_100.name} override'
print(wr10_100_measured1)

# port impedance at media construction method
wr10_meas = RectangularWaveguide(f_wr10, a=to_meters(100, 'mil'), b=to_meters(50, 'mil'), ep_r=1.0, rho=1.68e-08, z0_override=50)
wr10_100_measured2 = wr10_meas.line(100e-3, unit='m', name='wr10 100mm z0_port')
print(wr10_100_measured2)

# prepare figure
fig4, axes = plt.subplots(2, 2, figsize=(10, 6))
rf.stylely()
gs = axes[1, 0].get_gridspec()
for ax in axes[1, :]:
    ax.remove()
axbig = fig4.add_subplot(gs[1, :])

# plot return loss
wr10_100_measured1.plot_s_mag(0, 0, ax=axes[0, 0], label=wr10_100_measured1.name)
wr10_100_measured2.plot_s_mag(0, 0, ax=axes[0, 0], label=wr10_100_measured2.name)

# plot insertion loss
wr10_100_measured1.plot_s_db(1, 0, ax=axes[0, 1], label=wr10_100_measured1.name)
wr10_100_measured2.plot_s_db(1, 0, ax=axes[0, 1], label=wr10_100_measured2.name)

# plot port and characteristic impedances
axbig.plot(wr10_100_measured1.frequency.f_scaled, wr10_100_measured1.z0[:, 0].real,
           marker='d', markevery=30, label=f'line {wr10_100_measured1.name} z0')
axbig.plot(wr10_100_measured2.frequency.f_scaled, wr10_100_measured2.z0[:, 0].real,
           marker='x', markevery=30, label=f'line {wr10_100_measured2.name} z0')
axbig.plot(wr10.frequency.f_scaled, wr10.z0.real, label='media wr10 z0')
axbig.set_ylabel('Impedance (Ohm)')
# axbig.set_ylim(min(s1)-ss*0.1, max(s1)+ss*0.1)
axbig.set_xlabel(f'Frequency ({wr10.frequency.unit})')
axbig.legend(fontsize=6)

# resize plot nicely
fig4.tight_layout()
ssaver('./test/test_rf/test54.png')

# prepare
fig5, axes = plt.subplots(1, 2, figsize=(8, 3.5))
rf.stylely()
# plot
for ep_r in [4.0, 4.5, 5.0]:
    ml = MLine(f_mlin, w=3e-3, h=1.6e-3, t=35e-6, ep_r=ep_r, rho=1.68e-08, z0_port=50)
    axes[0].plot(f_mlin.f_scaled, ml.z0.real, label=f'ep_r={ep_r:.1f}')
    axes[1].plot(f_mlin.f_scaled, ml.beta, label=f'ep_r={ep_r:.1f}')

axes[0].set_xlabel(f'Frequency ({f_mlin.unit})')
axes[0].set_ylabel('Characteristic Impedance (Ohm)')
axes[0].legend()
axes[1].set_xlabel(f'Frequency ({f_mlin.unit})')
axes[1].set_ylabel('Propagation Constant (rad/m)')

axes[1].legend()

# resize plot nicely
fig5.tight_layout()
ssaver('./test/test_rf/test55.png')

freq = Frequency(10, 20, 101, 'GHz')
air = Freespace(freq)
pprint(88, air)

pprint(89, air.z0[:2])

# plane wave in Si
si = Freespace(freq, ep_r=11.2)
pprint(90, si.z0[:3])

slab = air.thru() ** si.line(1, unit='cm') ** air.thru()
slab.plot_s_db(n=0)
ssaver('./test/test_rf/test56.png')

pprint(91, mlin_meas.short(name='short'))
pprint(92, mlin_meas.line(d=90, unit='deg', name='line'))

delay_short = mlin_meas.line(d=90, unit='deg') ** mlin_meas.short()
delay_short.name = 'delay short'
pprint(93, delay_short)

tee = mlin_meas.tee()
delay_open = mlin_meas.delay_open(40, unit='deg')
shunt_open = rf.network.connect(tee, 1, delay_open, 0)
pprint(94, mlin_meas.shunt(delay_open))

delay_short = lambda d: mlin_meas.line(d, unit='deg') ** mlin_meas.short()  # noqa: E731
pprint(95, delay_short(90))

def shunt_stub(med, d0, d1):
    return med.line(d0, unit='deg')**med.shunt_delay_open(d1, unit='deg')


pprint(96, shunt_stub(mlin_meas, 10, 90))

freq = Frequency(75, 110, 101, 'GHz')
cpw = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6, z0_port=50)

# the load we are trying to match
load = cpw.load(.2+.2j)

# single stub circuit generator function
def shunt_stub(med, d0, d1):
    return med.line(d0, unit='deg') ** med.shunt_delay_open(d1, unit='deg')


# define the cost function we want to minimize (this uses sloppy namespace)
def cost(d):
    # prevent negative length lines, returning high cost
    if d[0] < 0 or d[1] < 0:
        return 1e3
    return (shunt_stub(cpw, d[0], d[1]) ** load)[100].s_mag.squeeze()


# initial guess of optimal delay lengths in degrees
d0 = 120, 40  # initial guess

# determine the optimal delays
d_opt = fmin(cost, (120, 40))

pprint(97, d_opt)

freq = rf.Frequency(1e-3, 10, 1001, 'ghz')
cpw = CPW(freq, w=0.6e-3, s=0.25e-3, ep_r=10.6, z0_port=50)

l1 = cpw.line(20, unit='mm')
c1 = cpw.shunt_capacitor(C=0.15e-12)
l1 = connect(c1, 1, l1, 0)
li = concat_ports([l1, l1], port_order='second')
Fix = li
Fix.name = 'Fix'
Fix.nudge(1e-4)
Left = Fix
# flip fixture for right side
Right = Fix.flipped()
l2 = cpw.line(50, 'mm')
DUT = concat_ports([l2, l2], port_order='second')
DUT.name = 'DUT'
DUT.nudge(1e-5)
Meas = Left ** DUT ** Right
Meas.name = 'Meas'
Meas.add_noise_polar(1e-5, 2)

DUTd = Left.inv ** Meas ** Right.inv
DUTd.name = 'DUTd'
fig, axarr = plt.subplots(2, 2, sharex=True, figsize=(10, 6))

ax = axarr[0, 0]
Meas.plot_s_db(m=0, n=0, ax=ax)
DUTd.plot_s_db(m=0, n=0, ax=ax)
DUT.plot_s_db(m=0, n=0, ax=ax, ls=':', color='0.0')
ax.set_title('Return Loss')
ax.legend(loc='lower center', ncol=3)
ax.grid(True)

ax = axarr[0, 1]
Meas.plot_s_db(m=2, n=0, ax=ax)
DUTd.plot_s_db(m=2, n=0, ax=ax)
DUT.plot_s_db(m=2, n=0, ax=ax, ls=':', color='0.0')
ax.set_title('Insertion Loss')
ax.legend(loc='lower center', ncol=3)
ax.grid(True)

ax = axarr[1, 0]
Meas.plot_s_db(m=1, n=0, ax=ax)
DUTd.plot_s_db(m=1, n=0, ax=ax)
DUT.plot_s_db(m=1, n=0, ax=ax, ls=':', color='0.0')
ax.set_title('Isolation')
ax.legend(loc='lower center', ncol=3)
ax.grid(True)

ax = axarr[1, 1]
Meas.plot_s_deg(m=2, n=0, ax=ax)
DUTd.plot_s_deg(m=2, n=0, ax=ax, marker='o', markevery=25)
DUT.plot_s_deg(m=2, n=0, ax=ax, ls=':', color='0.0')
ax.set_title('Insertion Loss')
ax.legend(loc='lower center', ncol=3)
ax.grid(True)

fig.tight_layout()
ssaver('./test/test_rf/test57.png')

# need this ideal s4p file to create a coax with a floating outer conductor
xformer = r"./test/test_rf/skrf/data/Ideal_4port_xformer_hi_res.s4p"
xformer_ntwk = rf.Network(xformer)
xformer_ntwk1 = xformer_ntwk["0.001-2ghz"]  # band center is thus 1 GHz

freq = xformer_ntwk1.frequency
z0_ports = 25

beta = freq.w / c
line_branch = rf.media.DefinedGammaZ0(frequency=freq, z0=z0_ports, gamma=0+beta*1j)

# d is 75 mm in length, which is quarterwave at 1 GHz in air
# d = line_branch.theta_2_d(90, deg=True, bc=True)
branch1 = line_branch.line(90, unit="deg", name="branch1")

lossy_line = line_branch.resistor(0.1, name="res")
port1 = Circuit.Port(freq, name="port1", z0=25)
port2 = Circuit.Port(freq, name="port2", z0=25)

# creating a 25-Ohm 2-port transmission line, with realistic insertion loss
# insertion loss is equivalent to a 0.1-Ohm series resistor
connections = [
    [(port1, 0), (lossy_line, 0)],
    [(lossy_line, 1), (branch1, 0)],
    [(branch1, 1), (port2, 0)],
]

C = Circuit(connections)

line1 = C.network
line1.name = "top line"
line2 = C.network
line2.name = "bottom line"

line = rf.media.DefinedGammaZ0(frequency=freq, z0=50)
ferrite_resistor = line.resistor(333, name="res")

port1a = Circuit.Port(frequency=freq, name="port1a", z0=50)
port2a = Circuit.Port(frequency=freq, name="port2a", z0=25)
port3a = Circuit.Port(frequency=freq, name="port3a", z0=25)
ground = Circuit.Ground(frequency=freq, name="ground", z0=50)

# creating a 4-port floating coax by feeding two 25-Ohm lines (twisted pair of wires)
# with an ideal transformer, to ensure the differential nature of the twisted pair, as per Fig.2
# the 333-Ohm resistor models the loading of the ferrite sleeve,
# upon the outer conductor of the 4-port coax line
conn = [
    [(port1a, 0), (xformer_ntwk1, 0)],
    [(xformer_ntwk1, 1), (line1, 0)],
    [(line1, 1), (port2a, 0)],
    [(ground, 0), (xformer_ntwk1, 2), (ferrite_resistor, 0)],
    [(xformer_ntwk1, 3), (line2, 0)],
    [(line2, 1), (ferrite_resistor, 1), (port3a, 0)],
]

C1 = Circuit(conn)
balun = C1.network
balun.name = "ideal balun"
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

balun.plot_s_db(ax=ax1, m=0, n=0, lw=2)  # S11
balun.plot_s_db(ax=ax1, m=1, n=1, lw=2)  # S22
ax1.set_ylim(-60, 0)

balun.plot_s_db(ax=ax2, m=1, n=0, lw=2)  # S21
ax2.set_ylim(-4, 0)

fig.suptitle("Ideal 50-Ohm balun")
ssaver('./test/test_rf/test58.png')

freq1 = rf.Frequency(start=0.01, stop=2, unit=('GHz'), npoints=2001)
beta = freq1.w / c

line1 = rf.media.DefinedGammaZ0(frequency=freq1, z0=50, gamma=0+beta*1j)

ferrite_resistor = line1.resistor(333, name='res')  # model ferrite sleeve as a parallel 333 Ohm resistor
Rp_circuit = ferrite_resistor

# define a floating 4-port transmission line, using a new element in ver 1.4 of Scikit-rf
line4p = line1.line_floating(90, unit='deg', z0=50, name='line4p')
port1 = Circuit.Port(frequency=freq1, name='port1', z0=50)
port2 = Circuit.Port(frequency=freq1, name='port2', z0=25)
port3 = Circuit.Port(frequency=freq1, name='port3', z0=25)
ground = Circuit.Ground(frequency=freq1, name='ground', z0=50)

connections = [[(port1, 0), (line4p, 0)],
               [(port2, 0), (line4p, 1)],
               [(port3, 0), (line4p, 3), (Rp_circuit, 0)],
               [(line4p, 2), (Rp_circuit, 1), (ground, 0)]]

circuit = Circuit(connections)
balun_circuit = circuit.network
balun_circuit.name = 'ideal balun circuit'

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

balun_circuit.plot_s_db(ax=ax1, m=0, n=0, lw=2)  # S11
balun_circuit.plot_s_db(ax=ax1, m=1, n=1, lw=2)  # S22
ax1.set_ylim(-60, 0)

balun_circuit.plot_s_db(ax=ax2, m=1, n=0, lw=2)  # S21
ax2.set_ylim(-4, 0)

fig.suptitle("Ideal 50-Ohm balun")
plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.1)

ssaver('./test/test_rf/test59.png')

freq = rf.Frequency(start=0.01, stop=2, unit=('GHz'), npoints=2001)
beta = freq.w/c

line = rf.media.DefinedGammaZ0(frequency=freq, z0=50, gamma=0+beta*1j)
ferrite_inductor = line.inductor(100e-9, name='ind')  # model ferrite sleeve as a parallel 100 nH inductor

# define a floating 4-port transmission line
line4p_a = line.line_floating(90, unit='deg', z0=50, name='line4p_a')
line4p_b = line.line_floating(90, unit='deg', z0=50, name='line4p_b')

port1 = Circuit.Port(frequency=freq, name='port1', z0=100)
port2 = Circuit.Port(frequency=freq, name='port2', z0=25)
ground1 = Circuit.Ground(frequency=freq, name='ground1', z0=50)
ground2 = Circuit.Ground(frequency=freq, name='ground2', z0=50)

connections = [[(port1, 0), (line4p_a, 0)],
               [(port2, 0), (line4p_a, 1), (line4p_b, 1)],
               [(line4p_a, 2), (line4p_b, 0), (ferrite_inductor, 0)],
               [(line4p_a, 3), (ferrite_inductor, 1), (ground1, 0), (line4p_b, 3)],
               [(line4p_b, 2), (ground2, 0)]]

circuit = Circuit(connections)
transf_circuit = circuit.network
transf_circuit.name = 'ideal Guanella transformer circuit'

transf_circuit.plot_s_db(m=0, n=0, lw=2)
transf_circuit.plot_s_db(m=1, n=0, lw=2)
transf_circuit.plot_s_db(m=1, n=1, lw=2)

plt.ylim(-50, 10)
ssaver('./test/test_rf/test60.png')

# ideal transformer to convert differential ports to 50-Ohms single-ended
xformer = r"/Users/user/Desktop/rust_project/rsplotlib/test/test_rf/skrf/data/Ideal_4port_xformer_hi_res.s4p"
xformer_ntwk = rf.Network(xformer)
xformer_ntwk1a = xformer_ntwk["0.001-2ghz"]

freq = xformer_ntwk1a.frequency
beta = freq.w/c

line = rf.media.DefinedGammaZ0(frequency=freq, z0=50, gamma=0+beta*1j)
C1 = line.capacitor(0.106e-12, name='cap')

# define a floating 4-port transmission line
line4p_a1 = line.line_floating(90, unit='deg', z0=50, name='line4p_a1')
line4p_b1 = line.line_floating(90, unit='deg', z0=120, name='line4p_b1')
line4p_c1 = line.line_floating(90, unit='deg', z0=10, name='line4p_c1')
line4p_d1 = line.line_floating(90, unit='deg', z0=120, name='line4p_d1')

port1a = Circuit.Port(frequency=freq, name='port1', z0=50)
port2a = Circuit.Port(frequency=freq, name='port2', z0=50)
ground1a = Circuit.Ground(frequency=freq, name='ground1', z0=50)
ground2a = Circuit.Ground(frequency=freq, name='ground2', z0=50)

connections = [[(port1a, 0), (line4p_a1, 0)],
               [(ground1a, 0), (line4p_a1, 2), (line4p_b1, 0), (line4p_b1, 2), (line4p_b1, 3)],
               [(line4p_a1, 1), (line4p_c1, 0)],
               [(line4p_a1, 3), (line4p_b1, 1), (xformer_ntwk1a, 0)],
               [(line4p_c1, 2), (xformer_ntwk1a, 2), (line4p_d1, 0)],
               [(line4p_d1, 1), (line4p_c1, 3), (line4p_d1, 2), (line4p_d1, 3), (C1, 1), (xformer_ntwk1a, 3), (ground2a, 0)],
               [(line4p_c1, 1), (C1, 0)],
               [(xformer_ntwk1a, 1), (port2a, 0)]]

circuit1 = Circuit(connections)
Marchand_circuit = circuit1.network
Marchand_circuit.name = 'Marchand circuit'

Marchand_circuit.plot_s_db(m=0, n=0, lw=2)
Marchand_circuit.plot_s_db(m=1, n=0, lw=2)
Marchand_circuit.plot_s_db(m=1, n=1, lw=2)

plt.ylim(-50, 10)
ssaver('./test/test_rf/test61.png')
