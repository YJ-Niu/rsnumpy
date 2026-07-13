import skrf as rf
from skrf.data import ring_slot  # noqa: F811
import rsnumpy as np
import rsplotlib.pyplot as plt


def pprint(ss):
    print("++++++++++++++++++++++++++++++")
    print(ss, "\n")


# ring_slot = rf.Network('data/ring slot.s2p')

# ring_slot

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
rf.stylely()
ring_slot.plot_s_db()
plt.savefig('/test/test_rf/ring_slot.png')
