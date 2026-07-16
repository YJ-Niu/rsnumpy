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

rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_db()
# ssaver('./test/test_rf/test1.png')


rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_deg(m=0, n=1)
# ssaver('./test/test_rf/test2.png')


rf.stylely(figsize=(20, 16), dpi=144)
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
except:
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

cir.plot_graph(network_labels=True, network_fontsize=15,
               port_labels=True, port_fontsize=15,
               edge_labels=True, edge_fontsize=10)

ssaver('./test/test_rf/test38.png')
