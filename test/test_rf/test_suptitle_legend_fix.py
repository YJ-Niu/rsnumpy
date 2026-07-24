import rsnumpy as np
import rsplotlib.pyplot as plt

freq = np.linspace(0.01, 2, 2001)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(freq, np.sin(freq * 10), lw=2, label='Signal 1')
ax1.plot(freq, np.cos(freq * 10), lw=2, label='Signal 2')
ax1.plot(freq, np.sin(freq * 5), lw=2, label='Signal 3')
ax1.set_ylim(-1, 1)

ax2.plot(freq, np.cos(freq * 10), lw=2, label='Signal A')
ax2.plot(freq, np.sin(freq * 5), lw=2, label='Signal B')
ax2.set_ylim(-1, 1)

fig.suptitle("Test Suptitle Position")
ax1.legend()
ax2.legend()

plt.savefig("./test/test_rf/suptitle_legend_test1.png")
print("Saved suptitle_legend_test1.png - suptitle at top, legends with auto layout")

plt.clf()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(freq, np.sin(freq * 10), lw=2, label='Signal 1')
ax.plot(freq, np.cos(freq * 10), lw=2, label='Signal 2')
ax.plot(freq, np.sin(freq * 5), lw=2, label='Signal 3')
ax.plot(freq, np.cos(freq * 5), lw=2, label='Signal 4')
fig.suptitle("Single Plot with Legend", pad=0)
ax.legend()
plt.savefig("./test/test_rf/suptitle_legend_test2.png")
print("Saved suptitle_legend_test2.png - single plot, pad=0")

plt.clf()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(freq, np.sin(freq * 10), lw=2, label='Signal 1')
ax.plot(freq, np.cos(freq * 10), lw=2, label='Signal 2')
fig.suptitle("Test with pad=10", pad=10)
ax.legend(loc='upper right')
plt.savefig("./test/test_rf/suptitle_legend_test3.png")
print("Saved suptitle_legend_test3.png - upper right legend")

plt.clf()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(freq, np.sin(freq * 10), lw=2, label='Signal 1')
ax.plot(freq, np.cos(freq * 10), lw=2, label='Signal 2')
ax.plot(freq, np.sin(freq * 5), lw=2, label='Signal 3')
ax.plot(freq, np.cos(freq * 5), lw=2, label='Signal 4')
ax.plot(freq, np.sin(freq * 2), lw=2, label='Signal 5')
fig.suptitle("Multiple Legend Entries")
ax.legend(loc='lower left')
plt.savefig("./test/test_rf/suptitle_legend_test4.png")
print("Saved suptitle_legend_test4.png - lower left with many entries")

plt.clf()

print("\nAll tests completed!")