import time
from dataclasses import dataclass

from mpi4py import MPI


@dataclass
class StepTimer:
    compute_time: float = 0.0
    comm_time: float = 0.0
    steps: int = 0
    _t0: float = 0.0
    _mode: str = ""

    def start_compute(self):
        self._t0 = time.time()
        self._mode = "compute"

    def end_compute(self):
        if self._mode == "compute":
            self.compute_time += time.time() - self._t0
            self.steps += 1
        self._mode = ""

    def start_comm(self):
        self._t0 = time.time()
        self._mode = "comm"

    def end_comm(self):
        if self._mode == "comm":
            self.comm_time += time.time() - self._t0
        self._mode = ""


def gather_metrics(comm, timer, local_agents):
    payload = {
        "compute": timer.compute_time,
        "comm": timer.comm_time,
        "steps": timer.steps,
        "agents": local_agents,
    }
    all_payloads = comm.gather(payload, root=0)
    if comm.rank == 0:
        total_agents = sum(p["agents"] for p in all_payloads)
        avg_compute = sum(p["compute"] for p in all_payloads) / len(all_payloads)
        avg_comm = sum(p["comm"] for p in all_payloads) / len(all_payloads)
        max_compute = max(p["compute"] for p in all_payloads)
        max_comm = max(p["comm"] for p in all_payloads)
        # print("[metrics] total_agents:", total_agents)
        # print("[metrics] avg_compute_s:", round(avg_compute, 4), "avg_comm_s:", round(avg_comm, 4))
        # print("[metrics] max_compute_s:", round(max_compute, 4), "max_comm_s:", round(max_comm, 4))
        return {
            "total_agents": total_agents,
            "avg_compute_s": avg_compute,
            "avg_comm_s": avg_comm,
            "max_compute_s": max_compute,
            "max_comm_s": max_comm,
            "steps": timer.steps,
        }
    return None
