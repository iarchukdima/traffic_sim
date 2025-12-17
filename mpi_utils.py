from mpi4py import MPI

def exchange_migrations(comm, outbound, neighbors = None, tag = 0):
    size = comm.size
    rank = comm.rank
    if neighbors is None:
        neighbors = [(rank - 1) % size, (rank + 1) % size]
    neighbors = list(set(neighbors))

    inbound = []
    recv_reqs = [comm.irecv(source=n, tag=tag) for n in neighbors]
    send_reqs = [comm.isend(outbound.get(n, []), dest=n, tag=tag) for n in neighbors]

    for req in recv_reqs:
        data = req.wait()
        if data:
            inbound.extend(data)
    for req in send_reqs:
        req.wait()
    return inbound

