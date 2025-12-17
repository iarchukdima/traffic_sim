import subprocess
import os
import numpy as np

def run_tests():
    script_name = "/Users/dmitrii/coding/hp_python/project/traffic_sim/main.py"

    if os.path.exists("result.txt"):
        os.remove("result.txt")

    sizes = [1, 2, 3, 4, 8, 12]

    args = ["--width", "1000", "--height", "1000", "--steps", "1000", "--agents", "3600", "--block", "40", "--snapshot-interval", "2000"]

    # Set PYTHONPATH in environment
    env = os.environ.copy()
    project_path = "/Users/dmitrii/coding/hp_python/project"
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{project_path}"
    else:
        env["PYTHONPATH"] = project_path

    # Try mpiexec first, fallback to mpirun
    mpi_cmd = "mpirun"

    for size in sizes:
        cmd = [mpi_cmd, "-n", str(size), "python3", "-m", "traffic_sim.main"] + args
        subprocess.run(cmd, env=env)

    #different map sizes
    map_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for map_size in map_sizes:
        args = ["--width", str(map_size), "--height", str(map_size)] + args[2:]
        for size in [8]:
            cmd = [mpi_cmd, "-n", str(size), "python3", "-m", "traffic_sim.main"] + args
            subprocess.run(cmd, env=env)

    #different number of agents
    for agents in [120, 240, 480, 960, 1920, 3840, 7680]:
        args = ["--width", "1000", "--height", "1000", "--steps", "1000", "--agents", str(agents), "--block", "40", "--snapshot-interval", "2000"]
        for size in [8]:
            cmd = [mpi_cmd, "-n", str(size), "python3", "-m", "traffic_sim.main"] + args
            subprocess.run(cmd, env=env)

if __name__ == "__main__":
    run_tests()