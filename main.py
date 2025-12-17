import argparse
import os
from typing import List, Tuple

from mpi4py import MPI

from traffic_sim.metrics import StepTimer, gather_metrics
from traffic_sim.model import TrafficModel
from traffic_sim.mpi_utils import exchange_migrations
from traffic_sim.viz import render_snapshot


def parse_args()
    parser = argparse.ArgumentParser(description="MPI traffic simulation")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--agents", type=int, default=500, help="agents per rank")
    parser.add_argument("--vmax", type=int, default=5)
    parser.add_argument("--p-slow", type=float, default=0.2)
    parser.add_argument("--p-turn", type=float, default=0.2)
    parser.add_argument("--block", type=int, default=10, help="road spacing (cells)")
    parser.add_argument("--lane-capacity", type=int, default=2, help="agents per direction per cell")
    parser.add_argument("--snapshot-interval", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="traffic_snapshots")
    parser.add_argument("--gif", action="store_true", help="save GIF from snapshots on rank 0")
    parser.add_argument("--gif-name", type=str, default="traffic.gif")
    parser.add_argument("--gif-fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def gather_snapshot(comm: MPI.Comm, local_agents: List[Tuple[int, int, str]]):
    all_agents = comm.gather(local_agents, root=0)
    if comm.rank == 0:
        merged: List[Tuple[int, int, str]] = []
        for part in all_agents:
            merged.extend(part)
        return merged
    return None


def main()
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    neighbors = [(rank - 1) % size, (rank + 1) % size]

    agents_per_rank = args.agents // size

    model = TrafficModel(
        comm=comm,
        width=args.width,
        height=args.height,
        vmax=args.vmax,
        p_slow=args.p_slow,
        p_turn=args.p_turn,
        agents_per_rank=agents_per_rank,
        seed=args.seed,
        block=args.block,
        lane_capacity=args.lane_capacity,
    )

    timer = StepTimer()
    snapshot_interval = args.snapshot_interval
    out_dir = args.output_dir

    if rank == 0:
        print(f"[init] ranks={comm.size} width={args.width} height={args.height} agents_per_rank={agents_per_rank}")

    for step in range(args.steps + 1):
        timer.start_compute()
        outbound = model.step()
        timer.end_compute()

        timer.start_comm()
        inbound = exchange_migrations(comm, outbound, neighbors)
        timer.end_comm()
        model.add_inbound(inbound)

        # if snapshot_interval > 0 and step % snapshot_interval == 0:
        #     snapshot = gather_snapshot(comm, model.snapshot())
        #     if rank == 0 and snapshot is not None:
        #         os.makedirs(out_dir, exist_ok=True)
        #         render_snapshot(step, snapshot, args.width, args.height, out_dir)
        #         print(f"[viz] saved snapshot for step {step}")

    metrics = gather_metrics(comm, timer, len(model.agents))

    if rank == 0 and args.gif:
        print(0)
        from traffic_sim.viz import save_gif  # defer import to rank 0

        # save_gif(out_dir, args.gif_name, fps=args.gif_fps)
        # print(f"[viz] saved GIF to {os.path.join(out_dir, args.gif_name)}")
        
    if rank == 0:
        with open("result.txt", "a") as f:
            f.write(f"{size},{metrics['avg_compute_s']},{metrics['avg_comm_s']}\n")
            

if __name__ == "__main__":
    main()


