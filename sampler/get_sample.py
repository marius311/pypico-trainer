#!/usr/bin/env python

"""
Prints to stdout a CAMB parameter ini file 
"""

from numpy import loadtxt, atleast_1d
import sys, os, re, random
from collections import defaultdict
from numpy.core.fromnumeric import cumsum


def read_samples(filename):
    """
    Reads a file where the first line is a comment listing the names of some parameters
    and the subsequent lines are sets of values for those parameters.
    
    Returns a list of dictionaries
    """
    with open(filename) as file:
        names = re.sub("#","",file.readline()).split()
        data = loadtxt(file)
        
    return [dict(zip(names,d)) for d in data]


def read_ini(file):
    """
    Read a dictionary from a file where each line is "key = value" or #comment
    """
    
    params = dict()
    
    with open(file) as f: lines=[l for l in [re.sub("#.*","",l).strip() for l in f.readlines()] if len(l)>0]
    for line in lines:
        tokens = [t.strip() for t in line.split("=")]
        if (len(tokens)!=2): raise SyntaxError("Error parsing "+file+". Expected one \"=\" in line '"+line+"'")
        else: params[tokens[0]] = tokens[1]

    return params


def write_ini(out,params,changedkeys,index):
    """
    Write the ini file for the updated parameters
    """
    
    out.write("#\n# The following parameters are set from sample "+str(index)+":\n#\n\n")
    for k in set(params.keys()) & set(changedkeys): out.write(str(k)+" = "+str(params[k])+'\n')
    out.write("\n\n#\n# The following parameters are at their default values:\n#\n\n") 
    for k in set(params.keys()) - set(changedkeys): out.write(str(k)+" = "+str(params[k])+'\n')
        
        
def get_sample_index(nsamp,started):
    """
    Randomly chooses one of the least started samples in [0,nsamp] 
    """
    if started:
        samp = [0]*nsamp
        for i in started: samp[i]+=1
        invsamp = defaultdict(lambda:[])
        for i,s in enumerate(samp): invsamp[s]+=[i]
        return random.choice(invsamp[min(invsamp.keys())])
    else:
        return 0
        
        
        
def get_sample(paramfile,samplefile,out):
    """
    Return a random row from samplefile with default values defined in paramfile
    
    Output is printed to out, and the function returns an integer corresponding to which
    row in samplefile was chosen.
    """
            
    #Read default parameters
    defs = read_ini(paramfile)
    
    #Read the samples
    samples = read_samples(samplefile)
    
    #Read the indices of the samples we've done so far
    startedfile = os.path.join(os.path.dirname(samplefile),"started_"+os.path.basename(samplefile))
    if os.path.exists(startedfile): started = list(atleast_1d(loadtxt(startedfile,dtype=int)))
    else: started = []
    
    #Get a new samples to do
    i = get_sample_index(len(samples),started)
    
    #Names of the output files
    outputs = {"scalar_output_file":"scal_%.7i" % i, 
               "tensor_output_file":"tens_%.7i" % i, 
               "lensed_output_file":"lens_%.7i" % i,
               "transfer_filename(1)": "trans_%.7i" % i}
    
    #Create and save the parameter file
    defs.update(samples[i])
    defs.update(outputs)
    write_ini(out,defs,set(samples[i].keys()) | set(outputs.keys()),i)
    
    #Mark that we've started a new sample
    with open(startedfile,'a') as f: f.write(str(i)+"\n")

    return i


if __name__=="__main__":
    
    with open("jobs") as f: jobs = [line.split() for line in f.readlines() if line.strip()]
    ps = (lambda ps: ps/ps[-1])(cumsum([float(j[0]) for j in jobs]))
    r = random.random()
    for (p,(_,pf,sf)) in zip(ps,jobs): 
        if (r<p):
            get_sample(pf,sf,sys.stdout)
            break
    
