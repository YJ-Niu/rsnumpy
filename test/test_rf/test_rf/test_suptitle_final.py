import rsnumpy as np
import rsplotlib.pyplot as plt

freq = np.linspace(0.01, 2, 2001)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(freq, np.sin(freq * 10), lw=2)
ax1.set_ylim(-1, 1)
ax1.set_title('S11')

ax2.plot(freq, np.cos(freq * 10), lw=2)
ax2.set_ylim(-1, 1)
ax2.set_title('S21')

fig.suptitle("Ideal 50-Ohm balun")
plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.1)

plt.savefig("./test/test_rf/suptitle_final_test.png")
print("Saved suptitle_final_test.png - suptitle should be at the top")

plt.clf()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(freq, np.sin(freq * 10), lw=2)
ax2.plot(freq, np.cos(freq * 10), lw=2)
fig.suptitle("Test with pad=0", pad=0)
plt.savefig("./test/test_rf/suptitle_pad0_test.png")
print("Saved suptitle_pad0_test.png - suptitle with pad=0")

plt.clf()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(freq, np.sin(freq * 10), lw=2)
ax2.plot(freq, np.cos(freq * 10), lw=2)
fig.suptitle("Test with pad=10", pad=10)
plt.savefig("./test/test_rf/suptitle_pad10_test.png")
print("Saved suptitle_pad10_test.png - suptitle with pad=10")

plt.clf()

print("\nAll tests completed!")