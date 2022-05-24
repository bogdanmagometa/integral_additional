import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mplcyberpunk

data = {
    20: 8937003,
    19: 8795713,
    18: 8911902,
    17: 9062211,
    16: 8987607,
    15: 9032987,
    14: 8953794,
    13: 8920337,
    12: 8583438,
    11: 9251631,
    10: 9767370,
    9: 10042964,
    8: 11172335,
    7: 11633218,
    6: 10936252,
    5: 12886175,
    4: 16077220,
    3: 21360577,
    2: 31964429,
    1: 63749776
}
data_concurrent_x = 0
data_concurrent_y = 66442290

plt.style.use("cyberpunk")

fig1 = plt.figure("Figure_1", figsize=(10, 5))
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(data.keys(), np.array(list(data.values())) / 10**6, marker='o', color='#4ded30')
ax1.scatter(0, data_concurrent_y / 10**6, color='red')
ax1.set_xlim(-0.11, 20.5)
ax1.set_ylim(-1.1, 70)
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.set_xlabel("Number of used thread objects")
ax1.set_ylabel("Time of execution, s")
ax1.legend(labels=["Multithreaded version (main thread + additional threads)",
                   'Single-threaded version (main thread only)'],
           loc=[0.4, 0.7])

mplcyberpunk.make_lines_glow()
#mplcyberpunk.add_underglow()

plt.savefig("./img/time_plot.png", dpi=100)
