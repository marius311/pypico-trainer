#!/usr/bin/env python

"""
This MPI script just repeatedly calls get_sample for samples and calls CAMB on them forever
"""

import sys, os, re
import camb4py
from mpi import *

if __name__=="__main__":
    
    if len(sys.argv)!=3:
        print 'Usage: sampler.py params/ output/'
        sys.exit()

    comm = MPI.COMM_WORLD
    (rank,size) = (comm.Get_rank(),comm.Get_size())

    params, output = sys.argv[1:]
    
    def work(i):
        print "Process %i is doing sample %i..." % (rank,i)
        res = camb(**camb4py.read_ini(os.path.join(params,'params_%.7i'%i)))
        with open(os.path.join(output,'result_%.7i'),'w') as f: cPickle.dump(f,res,protocol=2)

    def getidxs(folder):
        set(int(re.search('([0-9]+)',f).group(0)) for f in os.listdir(folder))
        
    allidx = getidxs(params)
    idx = allidx - getidxs(output)
    if rank==0: print 'Doing %i of %i samples...'%(len(idxs),len(allidx))
    mpi_map(work,list(sorted(idxs)))



