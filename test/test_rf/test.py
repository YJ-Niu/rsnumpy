import skrf as rf
from skrf.data import ring_slot  # noqa: F811
import rsnumpy as np
import rsplotlib.pyplot as plt
from skrf.networkSet import NetworkSet
from skrf.media import CPW
from skrf import Frequency

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
plt.savefig('./test/test_rf/ring_slot.png')
plt.clf()

rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_deg(m=0, n=1)
plt.savefig('./test/test_rf/ring_slot_deg.png')
plt.clf()

rf.stylely(figsize=(20, 16), dpi=144)
ring_slot.plot_s_smith(lw=2)
plt.title('Big ole Smith Chart')
plt.savefig('./test/test_rf/ring_slot_smith.png')
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
plt.savefig('./test/test_rf/ro_uncertainty_bounds.png')
plt.clf()

freq = Frequency(75, 110, 101, 'GHz')
cpw = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)
pprint(cpw)
