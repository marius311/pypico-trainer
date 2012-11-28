#!/usr/bin/env python

"""
This MPI script just repeatedly calls get_sample for samples and calls CAMB on them forever
"""

import sys, os, re
from cStringIO import StringIO
from get_sample import get_sample
from mpi4py import MPI
from subprocess import call, check_output
from numpy import loadtxt
import tempfile

camb = '/global/scratch/sd/planck/user/marius/pico3/train/camb/camb'


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


if __name__=="__main__":
    
    if len(sys.argv)!=4:
        print 'Usage: sampler.py params.ini cps.dat output'
        sys.exit()

    comm = MPI.COMM_WORLD
    (rank,size) = (comm.Get_rank(),comm.Get_size())

    params, cps, output = sys.argv[1:]
    
    def work(i):
        work_filename = os.path.join(output,'params_%.7i'%i)
        with open(work_filename,'w') as work_file:
            get_sample(params,cps,work_file,i,output_prefix=os.path.abspath(output))
        print "Process %i is doing sample %i..." % (rank,i)
        check_output([camb,work_filename])
        #Reduce files a bit...?

    num_cps = len(loadtxt(cps))
    idxs = set(range(num_cps))-set(int(re.search('([0-9]+)',f).group(0)) for f in os.listdir(output))
    if rank==0: print 'Doing %i of %i samples...'%(len(idxs),num_cps)
    mpi_map(work,list(sorted(idxs)))



