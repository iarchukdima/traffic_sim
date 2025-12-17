import matplotlib.pyplot as plt
import numpy as np
import os

def make_plots():
    if not os.path.exists("result.txt"):
        raise FileNotFoundError("result.txt not found")

    with open("result.txt", "r") as f:
        data = f.readlines()

    sizes = [int(line.split(",")[0]) for line in data]
    data1lines = [float(line.split(",")[1]) for line in data]
    data2lines = [float(line.split(",")[2]) for line in data]

    # first

    sizes1 = sizes[:6]
    data1 = data1lines[:6]
    data2 = data2lines[:6]

    plt.plot(sizes1, data1, label="Compute Time")
    plt.plot(sizes1, data2, label="Communication Time")
    plt.xlabel("Size")
    plt.ylabel("Time")
    plt.title("Time vs Size")
    plt.legend()
    plt.savefig("/Users/dmitrii/coding/hp_python/project/traffic_sim/plots/result_compute1.png")
    plt.close()


    # second
    sizes2 = sizes[6:16]
    data1 = data1lines[6:16]
    data2 = data2lines[6:16]

    width = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    plt.plot(width, data1, label="Compute Time")
    plt.plot(width, data2, label="Communication Time")
    plt.xlabel("Width")
    plt.ylabel("Time")
    plt.title("Time vs Width")
    plt.legend()
    plt.savefig("/Users/dmitrii/coding/hp_python/project/traffic_sim/plots/result_compute2.png")
    plt.close()

    # third
    sizes3 = sizes[16:]
    data1 = data1lines[16:]
    data2 = data2lines[16:]

    agents = [120, 240, 480, 960, 1920, 3840, 7680]

    plt.plot(agents, data1, label="Compute Time")
    plt.plot(agents, data2, label="Communication Time")
    plt.xlabel("Agents")
    plt.ylabel("Time")
    plt.title("Time vs Agents")
    plt.legend()
    plt.savefig("/Users/dmitrii/coding/hp_python/project/traffic_sim/plots/result_compute3.png")
    plt.close()

if __name__ == "__main__":
    make_plots()