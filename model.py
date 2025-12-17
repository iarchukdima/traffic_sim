import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, DefaultDict
from collections import defaultdict

from mpi4py import MPI


@dataclass
class Agent:
    agent_id: int
    x: int
    y: int
    direction: str
    speed: int


def partition_bounds(rank: int, size: int, height: int) -> Tuple[int, int]:
    base = height // size
    extra = height % size
    start = rank * base + min(rank, extra)
    rows = base + (1 if rank < extra else 0)
    end = start + rows
    return start, end


def build_roads(width: int, height: int, block: int) -> List[List[Set[str]]]:
    allowed: List[List[Set[str]]] = [[set() for _ in range(height)] for _ in range(width)]
    # vertical one-way roads
    for idx, x in enumerate(range(0, width, block)):
        direction = "N" if idx % 2 == 0 else "S"
        for y in range(height):
            allowed[x][y].add(direction)
    # horizontal one-way roads
    for idx, y in enumerate(range(0, height, block)):
        direction = "E" if idx % 2 == 0 else "W"
        for x in range(width):
            allowed[x][y].add(direction)
    return allowed


def is_intersection(road: List[List[Set[str]]], x: int, y: int) -> bool:
    if not road[x][y]:
        return False
    # intersection when both horizontal and vertical directions exist
    has_h = any(d in ("E", "W") for d in road[x][y])
    has_v = any(d in ("N", "S") for d in road[x][y])
    return has_h and has_v


class TrafficModel:
    def __init__(
        self,
        comm: MPI.Comm,
        width: int,
        height: int,
        vmax: int,
        p_slow: float,
        p_turn: float,
        agents_per_rank: int,
        seed: int,
        block: int = 10,
        lane_capacity: int = 2,
    ):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.width = width
        self.height = height
        self.vmax = vmax
        self.p_slow = p_slow
        self.p_turn = p_turn
        self.rng = random.Random(seed + self.rank)
        self.road = build_roads(width, height, block)
        self.lane_capacity = lane_capacity
        self.occ: DefaultDict[Tuple[int, int, str], int] = defaultdict(int)
        self.y_start, self.y_end = partition_bounds(self.rank, self.size, self.height)
        self.partitions = [partition_bounds(r, self.size, self.height) for r in range(self.size)]
        self.neighbors = [r for r in (self.rank - 1, self.rank + 1) if 0 <= r < self.size]
        self._next_id = self.rank * 1_000_000
        self.agents: List[Agent] = []
        self._init_agents_on_roads(agents_per_rank)
        self._rebuild_occupancy()

    def _init_agents_on_roads(self, count: int) -> None:
        road_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.y_start, self.y_end)
            if self.road[x][y]
        ]
        for _ in range(count):
            x, y = self.rng.choice(road_cells)
            direction = self._pick_valid_direction(x, y)
            speed = self.rng.randrange(self.vmax + 1)
            self.agents.append(Agent(self._alloc_id(), x, y, direction, speed))

    def _alloc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def step(self) -> Dict[int, List[Dict]]:
        outbound: Dict[int, List[Dict]] = {}
        next_local: List[Agent] = []
        # rebuild occupancy at start of step
        self._rebuild_occupancy()

        for agent in self.agents:
            updated, dest_rank = self._advance_agent(agent)
            if dest_rank == self.rank:
                next_local.append(updated)
            else:
                outbound.setdefault(dest_rank, []).append(self._serialize(updated))

        self.agents = next_local
        return outbound

    def _advance_agent(self, agent: Agent) -> Tuple[Agent, int]:
        # free current lane before moving
        self._dec_occ(agent.x, agent.y, agent.direction)
        # Turn only at intersections
        if is_intersection(self.road, agent.x, agent.y) and self.rng.random() < self.p_turn:
            candidates = [d for d in ("N", "S", "E", "W") if self._road_neighbor(agent.x, agent.y, d)]
            if candidates:
                agent.direction = self.rng.choice(candidates)

        # Accelerate
        if agent.speed < self.vmax:
            agent.speed += 1

        # Random slow-down
        if self.rng.random() < self.p_slow:
            agent.speed = max(0, agent.speed - 1)

        steps = agent.speed
        x, y = agent.x, agent.y

        while steps > 0:
            dx, dy = self._direction_vector(agent.direction)
            nx, ny = self._wrap_coords(x + dx, y + dy)
            if self._road_neighbor(x, y, agent.direction) and self._can_enter(nx, ny, agent.direction):
                x, y = nx, ny
                steps -= 1
            else:
                # try to turn when blocked (e.g., at grid bound or lane full)
                turn_candidates = [
                    d
                    for d in ("N", "S", "E", "W")
                    if d != agent.direction and self._road_neighbor(x, y, d)
                    and self._can_enter(*self._wrap_coords(x + self._direction_vector(d)[0], y + self._direction_vector(d)[1]), d)
                ]
                if turn_candidates:
                    agent.direction = self.rng.choice(turn_candidates)
                    continue
                agent.speed = 0
                break

        agent.x, agent.y = x, y
        self._inc_occ(agent.x, agent.y, agent.direction)
        dest_rank = self.rank_for_y(agent.y)
        return agent, dest_rank

    def add_inbound(self, inbound: List[Dict]) -> None:
        for payload in inbound:
            self.agents.append(self._deserialize(payload))
        # resolve cross-rank collisions by enforcing lane_capacity on arrivals
        if inbound:
            self._resolve_collisions()

    def snapshot(self) -> List[Tuple[int, int, str]]:
        return [(a.x, a.y, a.direction) for a in self.agents]

    def rank_for_y(self, y: int) -> int:
        for r, (start, end) in enumerate(self.partitions):
            if start <= y < end:
                return r
        return self.size - 1

    def _pick_valid_direction(self, x: int, y: int) -> str:
        candidates = [d for d in ("N", "S", "E", "W") if self._road_neighbor(x, y, d)]
        return self.rng.choice(candidates) if candidates else "N"

    def _road_neighbor(self, x: int, y: int, direction: str) -> bool:
        # Direction must be allowed to depart from current cell
        if direction not in self.road[x][y]:
            return False
        dx, dy = self._direction_vector(direction)
        nx, ny = self._wrap_coords(x + dx, y + dy)
        return bool(self.road[nx][ny])

    def _can_enter(self, x: int, y: int, direction: str) -> bool:
        return self.occ[(x, y, direction)] < self.lane_capacity

    def _inc_occ(self, x: int, y: int, direction: str) -> None:
        self.occ[(x, y, direction)] += 1

    def _dec_occ(self, x: int, y: int, direction: str) -> None:
        key = (x, y, direction)
        if self.occ.get(key, 0) > 0:
            self.occ[key] -= 1

    def _rebuild_occupancy(self) -> None:
        self.occ = defaultdict(int)
        for a in self.agents:
            self._inc_occ(a.x, a.y, a.direction)

    def _wrap_coords(self, x: int, y: int) -> Tuple[int, int]:
        return x % self.width, y % self.height

    def _resolve_collisions(self) -> None:
        """Redistribute arrivals without dropping: try neighbors, else keep crowded."""
        buckets: DefaultDict[Tuple[int, int, str], List[Agent]] = defaultdict(list)
        for a in self.agents:
            buckets[(a.x, a.y, a.direction)].append(a)

        survivors: List[Agent] = []
        for (x, y, d), group in buckets.items():
            if len(group) <= self.lane_capacity:
                survivors.extend(group)
                continue

            survivors.extend(group[: self.lane_capacity])
            for extra in group[self.lane_capacity:]:
                placed = False
                for alt_d in ("N", "S", "E", "W"):
                    dx, dy = self._direction_vector(alt_d)
                    nx, ny = self._wrap_coords(x + dx, y + dy)
                    if alt_d in self.road[x][y] and self._can_enter(nx, ny, alt_d):
                        extra.x, extra.y, extra.direction = nx, ny, alt_d
                        survivors.append(extra)
                        placed = True
                        break
                if not placed:
                    survivors.append(extra)

        self.agents = survivors
        self._rebuild_occupancy()

    @staticmethod
    def _direction_vector(direction: str) -> Tuple[int, int]:
        return {
            "N": (0, 1),
            "S": (0, -1),
            "E": (1, 0),
            "W": (-1, 0),
        }.get(direction, (0, 0))

    @staticmethod
    def _turn(direction: str) -> str:
        if direction in ("N", "S"):
            return random.choice(["E", "W"])
        return random.choice(["N", "S"])

    def _serialize(self, agent: Agent) -> Dict:
        return {
            "id": agent.agent_id,
            "x": agent.x,
            "y": agent.y,
            "d": agent.direction,
            "v": agent.speed,
        }

    def _deserialize(self, payload: Dict) -> Agent:
        return Agent(
            agent_id=payload["id"],
            x=payload["x"],
            y=payload["y"],
            direction=payload["d"],
            speed=payload["v"],
        )


