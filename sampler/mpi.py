from mpi4py import MPI

def mpi_map(function,sequence,distribute=True):
    """
    A map function parallelized with MPI. If this program was called with mpiexec -n $NUM, 
    then partitions the sequence into $NUM blocks and each MPI process does the rank-th one.
    Note: If this function is called recursively, only the first call will be parallelized

    Keyword arguments:
    distribute -- If true, every process receives the answer
                  otherwise only the root process does (default=True)
    """
    comm = MPI.COMM_WORLD
    (rank,size) = (comm.Get_rank(),comm.Get_size())

    if (size==1): return map(function,sequence)
    else: return flatten(comm.allgather(map(function, partition(sequence,size)[rank])))


def flatten(l):
    """Returns a list of lists joined into one"""
    return list(itertools.chain(*l))

def partition(list, n):
    """Partition list into n nearly equal sublists"""
    division = len(list) / float(n)
    return [list[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]
