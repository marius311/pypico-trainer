#!/usr/bin/env python

data_dir = '/global/project/projectdirs/planck/user/marius/uspype/wmap7/data/'



import sys
import cosmoslik.plugins.likelihoods.wmap as W
from cosmoslik.plugins.likelihoods.wmap.pywmap import pywmap
from scipy.interpolate import splrep, splev
import cPickle, itertools
from numpy import arange
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

    def flatten(l):
        """Returns a list of lists joined into one"""
        return list(itertools.chain(*l))

    def partition(list, n):
        """Partition list into n nearly equal sublists"""
        division = len(list) / float(n)
        return [list[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

    comm = MPI.COMM_WORLD
    (rank,size) = (comm.Get_rank(),comm.Get_size())

    if (size==1): return map(function,sequence)
    else: return flatten(comm.allgather(map(function, partition(sequence,size)[rank])))



if __name__=='__main__':

    if len(sys.argv)<1:
        print 'Usage: sampler_wmap.py input.dat "dict(..)" key_name output.dat'

    with open(sys.argv[1]) as f: dat = cPickle.load(f)

    initargs = eval(sys.argv[2])

    pywmap.wmapinit(data_dir=data_dir,**initargs)

    def ipl(d): return splev(arange(2,2000),splrep(*d.T))

    def work(k):
        d = dat[k]
        lnl = sum(pywmap.wmaplnlike(cltt=ipl(d['lens'][:,[0,1]]),
                                    clte=ipl(d['lens'][:,[0,4]]),
                                    clee=ipl(d['lens'][:,[0,2]]),
                                    clbb=ipl(d['lens'][:,[0,3]])))
        print "Process %i got %f for %s"%(MPI.COMM_WORLD.Get_rank(),lnl,d['cps'])
        return (k,{sys.argv[3]:lnl})
    
    wdats = mpi_map(work,[k for k,v in dat.iteritems() if 'lens' in v],distribute=False)

    if MPI.COMM_WORLD.Get_rank()==0:
        for k,v in wdats: dat[k].update(v)
        with open(sys.argv[4],'w') as f: cPickle.dump(dat,f,protocol=2)
