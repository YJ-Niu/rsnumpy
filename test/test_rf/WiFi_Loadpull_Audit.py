import rsplotlib.pyplot as plt
# import rsnumpy as np
from rsplotlib import style

import skrf as rf

# 保存当前图
def ssaver(name):
    plt.savefig(name)
    plt.clf()


s2p = rf.Network('/Users/user/Downloads/WI-FI loadpull-APL/New cable/Audit_5G_New_cable1.s2p')
s2p.frequency.unit = 'ghz'
mpl_style = "seaborn-ticks"
with style.context(mpl_style):
    s2p.plot_s_smith(m=0, n=0, draw_labels=True)
plt.legend()  # have to re-generate legend

ssaver('./test/test_rf/test_1.png')

s2p = rf.Network('/Users/user/Downloads/WI-FI loadpull-APL/New cable/Audit_5G_New_cable2.s2p')
s2p.frequency.unit = 'ghz'
mpl_style = "seaborn-ticks"
with style.context(mpl_style):
    s2p.plot_s_smith(m=0, n=0, draw_labels=True)
plt.legend()  # have to re-generate legend

ssaver('./test/test_rf/test_2.png')

s2p = rf.Network('/Users/user/Downloads/WI-FI loadpull-APL/New cable/Audit_5G_Old_cable.s2p')
s2p.frequency.unit = 'ghz'
mpl_style = "seaborn-ticks"
with style.context(mpl_style):
    s2p.plot_s_smith(m=0, n=0, draw_labels=True)
plt.legend()  # have to re-generate legend

ssaver('./test/test_rf/test_3.png')
