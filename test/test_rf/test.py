import skrf as rf
from skrf import Frequency, Network
from skrf.data import ring_slot  # noqa: F811
import rsnumpy as np
import rsplotlib.pyplot as plt
from skrf.networkSet import NetworkSet
from skrf.media import CPW, Coaxial
from skrf.data import wr2p2_line1 as line1
from skrf.data import wr1p5_line, wr2p2_line

def pprint(ss):
    print("++++++++++++++++++++++++++++++")
    print(ss, "\n")


# ring_slot = rf.Network('data/ring slot.s2p')

# ring_slot
pprint(ring_slot)
short = rf.data.wr2p2_short
delayshort = rf.data.wr2p2_delayshort

pprint(short - delayshort)
pprint(short/delayshort)

short = rf.data.wr2p2_short
line = rf.data.wr2p2_line

delayshort = line ** short
short = line.inv ** delayshort
pprint(short)
print(type(line.s))
print(line.s.shape)

print(line.frequency)
pprint(line.f[0:10])

rs = rf.data.ring_slot  # another 2-port example
pprint(rs.s_mag[:, 1, 0].min())
f_match = rs.f[np.argmin(rs.s_mag[:, 0, 0])]  # frequency for min(|S11|)
pprint(f_match)

rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_db()
plt.savefig('./test/test_rf/test1.png')
plt.clf()

rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_deg(m=0, n=1)
plt.savefig('./test/test_rf/test2.png')
plt.clf()

rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_smith(lw=2)
plt.title('Big ole Smith Chart')
plt.savefig('./test/test_rf/test3.png')
plt.clf()

print(rf.io.read_all('./test/test_rf/skrf/data/', contains='ro'))
ro_dict = rf.io.read_all('./test/test_rf/skrf/data/', contains='ro')
ro_ns = NetworkSet(ro_dict, name='ro set')  # name is optional
print(ro_ns)
pprint(ro_ns.mean_s)
pprint(ro_ns.std_s)

ro_ns.std_s.plot_s_mag(label='S11')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of RO')
plt.legend()
plt.savefig('./test/test_rf/ro_std_s.png')
plt.clf()

ro_ns.plot_uncertainty_bounds_s_db(label='S11')
plt.savefig('./test/test_rf/test4.png')
plt.clf()

freq = Frequency(75, 110, 101, 'GHz')
cpw = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)
pprint(cpw)

pprint(cpw.line(d=90, unit='deg', name='line'))

freq = Frequency(1, 10, 101, 'GHz')
coax = Coaxial(frequency=freq, Dint=1e-3, Dout=2e-3)
pprint(coax)

# dummy 2-port network from Frequency and s-parameters
freq = Frequency(1, 10, 101, 'ghz')
rng = np.random.default_rng()
s = rng.uniform(size=(101, 2, 2)) + 1j*rng.uniform(size=(101, 2, 2))  # random complex numbers
# if not passed, will assume z0=50. name is optional but it's a good practice.
ntwk = Network(frequency=freq, s=s, name='random values 2-port')
pprint(ntwk)

ntwk.plot_s_db()
plt.savefig('./test/test_rf/test5.png')
plt.clf()

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

pprint(ntw)

ntw2 = rf.Network(frequency=freq, s=s, z0=25, name='same z0 for all ports')
pprint(ntw2)
ntw3 = rf.Network(frequency=freq, s=s, z0=[20, 30], name='different z0 for each port')
pprint(ntw3)
ntw4 = rf.Network(frequency=freq, s=s, z0=rng.uniform(size=(4, 2)), name='different z0 for each frequencies and ports')
pprint(ntw4)

# 1-port network example
z = np.full((len(freq), 1, 1), 10j)  # replicate z=10j for all frequencies

ntw = rf.Network(frequency=freq, z=z)
pprint(ntw)

z = 20
abcd = np.array([[1, z], [0, 1]])

a = np.tile(abcd, (len(freq), 1, 1))
ntw = Network(frequency=freq, a=a)
pprint(ntw)

# example: converting a -> s
s = rf.network.a2s(a)
# checking that these S-params are the same
pprint(np.all(ntw.s == s))

pprint(np.shape(ring_slot.s))

s_a = ring_slot.s[:11, 1, 0]  # get first 10 values of S21
pprint(s_a)
pprint(ring_slot[0:10])

pprint(ring_slot['80-90ghz'])

pprint(ring_slot.s11['80-90ghz'])

rf.stylely()
ring_slot.plot_s_smith()
plt.savefig('./test/test_rf/test6.png')
plt.clf()

plt.title('Ring Slot $S_{21}$')

rf.stylely()
ring_slot.s11.plot_s_db(label='Full Band Response')
ring_slot.s11['82-90ghz'].plot_s_db(lw=3, label='Band of Interest')
plt.legend()
plt.savefig('./test/test_rf/test7.png')
plt.clf()


short - delayshort
short + delayshort
short * delayshort
pprint(short / delayshort)

difference = (short - delayshort)
difference.plot_s_mag(label='Mag of difference')
plt.savefig('./test/test_rf/test8.png')
plt.clf()

(delayshort/short).plot_s_deg(label='Detrended Phase')
plt.savefig('./test/test_rf/test9.png')
plt.clf()

hopen = (short*-1)
pprint(hopen.s[:3, ...])

rando = hopen * rng.uniform(size=len(hopen))
pprint(rando.s[:3, ...])

pprint(short == delayshort)
pprint(short != delayshort)

short = rf.data.wr2p2_short
line = rf.data.wr2p2_line
delayshort = line ** short
short_2 = line.inv ** delayshort

pprint(short_2 == short)

tee = rf.data.tee
pprint(tee)

terminated_tee = rf.network.connect(tee, 1, delayshort, 0)
pprint(terminated_tee)

pprint(line)
pprint(line1)
line1.resample(201)
pprint(line1)

line1 + line


big_line = rf.network.stitch(wr2p2_line, wr1p5_line)
pprint(big_line)
pprint(wr2p2_line)
pprint(wr1p5_line)
