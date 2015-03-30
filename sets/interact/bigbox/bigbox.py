
from numpy import *
from numpy.linalg import *
from pypico import CantUsePICO, PICO, create_pico
import os, re
from math import factorial
from scipy.interpolate import splrep, splev
from collections import defaultdict
from scipy.linalg import solve
from itertools import chain
import cPickle as pickle


def getpoly(names,order):
    if type(order)==int: order = dict((n,order) for n in names)
    maxorder = max(order.values())
    
    def getterms(names,maxorder):
        if maxorder==0 or len(names)==0: return [[]]
        else:
            return [([] if i==0 else [(names[0],i)])+p 
                    for i in range(0,min(maxorder,order[names[0]])+1) 
                    for p in getterms(names[1:],maxorder-i)]

    terms = getterms(names,maxorder)
    code = "lambda p: ["+",".join(["*".join(["p['%s'][%i]"%t for t in term]) for term in terms[1:]])+"]"
    evalpoly = eval(code)
        
    def poly(params,derive=None):
        zr, os = params[names[0]]**-inf, params[names[0]]**0
        d = derive if derive!=None else dict((n,0) for n in names)
        
        polys = dict((n,[zr if d[n]>i else factorial(i)/factorial(i-d[n])*params[n]**(i-d[n]) 
                    for i in range(order[n]+1)]) for n in names)
        
        return [os]+evalpoly(polys)
    
    poly.terms = terms  
    poly.code = code
    poly.names = names
    poly.order = order
    poly.__reduce_ex__ = lambda _: (getpoly,(names,order)) #Here's how to pickle this sucker
    
    return poly


def get_pico():
    """Returns the pico object for this dataset"""
    
    from pandas import DataFrame
    ntest = 2000

    data = {'fits':{}}
    
    lvals=loadtxt("lvalues",dtype=int)[:-125]
    
        
    dat=pickle.load(open("samples.dat"))
    ps=DataFrame([d[0] for d in dat])
    cls=array([d[1][lvals]*lvals*(lvals+1)/2/pi*(2.72e6)**2 for d in dat])
    goodidx=array([i for i,cl in enumerate(cls) if not any(isnan(cl)) and all(cl>0)])
    ps=ps.ix[goodidx]
    cls=cls[goodidx]
    psd={k:array(ps[k].tolist()) for k in ps}        
    fit=PicoFit(SameOrderInputter(3,psd),DefaultOutputter(psd,cls,lvals,arange(3000)))
    
    data['fits']['lensed_TT']=[fit]
    
    pico = tailmonty(data)
    
    return pico




class tailmonty(PICO):
    
    def __init__(self,data):
        self._data = data
        
        
    def get(self, outputs=None, force=False, **inputs):
        """
        Run PICO to get some outputs given some set of inputs.
        
        Parameters
        ----------
        
        outputs, optional : A list of outputs which to actually get. You can speed up 
                            computation time by only getting those outputs you need.
                            Valid outputs are given by PICO.outputs(). 
                            (default: all of them)
                            
        **inputs : A dictionary of the values of the inputs. Valid input names 
                   are given by PICO.inputs() and an example set of inputs
                   is given by PICO.example_inputs()
        """
        result = {}
        
        outputs = set(outputs or self.outputs())
        
        if 'k' in outputs:
            outputs.remove('k')
            result['k'] = self._data['k']
        
        xs = [m.group(1) for m in [re.match('cl_([TEB]{2})\Z',o) for o in outputs] if m!=None]
        Alens = inputs.get('Alens',1)
        for x in xs:
            if Alens!=1 and x!='BB': outputs.add('scalar_%s'%x)
            if Alens!=0: outputs.add('lensed_%s'%x)
        
        for output in outputs:
            if not output in ['cl_%s'%x for x in ['TT','TE','EE','BB']]:
                for region in self._data['fits'][output]:
                    try:
                        r = region.get_result(inputs,force=force)
                        result[output] = r
                        break
                    except CantUsePICO as e: 
                        pass
                else: 
                    raise e
        
        for x in xs:
            r = 0
            if Alens!=1 and x!='BB': 
                r += (1-Alens) * result['scalar_%s'%x]
            if Alens!=0:
                r += Alens * result['lensed_%s'%x]
            result['cl_%s'%x] = atleast_1d(r)

        return result
    
    
    def inputs(self):
        return set(chain(['Alens'],*(region.inputter.inputs() for fit in self._data['fits'].values() for region in fit)))  
        
    def outputs(self):
        return ['cl_TT','cl_TE','cl_EE','cl_BB','k']+self._data['fits'].keys()

    def example_inputs(self):
        """
        An example set of inputs, which can be passed to pico.get(**pico.example_inputs())
        """
        return {'helium_fraction': 0.248,
                'massive_neutrinos': 3.046,
                'ombh2': 0.022,
                'omch2': 0.125,
                'omnuh2': 0,
                're_optical_depth': 0.085,
                'scalar_amp(1)': 2.5e-09,
                'scalar_nrun(1)': 0,
                'scalar_spectral_index(1)': 0.97,
                'theta': 0.0104}
        

class Cleanable(object):
    def __getstate__(self):
        """Get rid of things which don't need to be stored in the datafile. """
        state = self.__dict__
        if hasattr(self, 'get_garbage'):
            for a in self.get_garbage(): state.pop(a,None)
        return state
            

class DefaultInputter(Cleanable):

    def __init__(self, inputs):
        self.inputs = inputs
        self.xinputs = self.inputs_to_xinputs(**inputs)
        self.xinput_center = self.get_xinput_center(self.xinputs)
        self.poly = getpoly(self.xinputs_order().keys(),self.xinputs_order())
        self.coefficients = self.xinputs_to_coeffs(self.xinputs)
        self.qr = qr(array(self.coefficients))
        self.input_center = self.get_xinput_center(self.xinputs)
        self.calc_bounds()
                 
                 
    def inputs(self):
        return ['Omega_k',
                 'A_s',
                 '100*theta_s',
                 'N_ur',
                 'omega_b',
                 'YHe',
                 'n_s',
                 'tau_reio',
                 'omega_cdm']
    
    
    def inputs_to_xinputs(self,**inputs):
#         fnu = 7./8*(4./11)**4./3 * inputs['massive_neutrinos']
#         inputs['zeq'] = (inputs['ombh2']+inputs['omch2']+inputs['omnuh2'])/(1+fnu)
#         inputs['thetad'] = (1+fnu)**.22/sqrt(1-inputs['helium_fraction'])
#         inputs['logA'] = log(10**10 * inputs['scalar_amp(1)'])
        return inputs
    
    def get_xinput_center(self,xinputs):
        return {k:mean(v) for k,v in xinputs.items()}
    
    
    def xinputs_to_coeffs(self,xinputs):
        """Given a set of xinputs, get the terms of the fitting polynomial."""
        return array(self.poly({k:(xinputs[k]-self.xinput_center[k]) for k in self.xinputs_order().keys()})).T
    
    
    def calc_bounds(self):
        self.xinput_region_center = {k:mean(self.xinputs[k]) for k in self.xinputs_order()}
        xinput_matrix = vstack([self.xinputs[k] - self.xinput_region_center[k] for k in self.xinputs_order()])
        icov = inv(cov(xinput_matrix))
        self.xinput_region_icov = icov / max(sqrt(sum(xinput_matrix*dot(icov,xinput_matrix),axis=0)))**2
        self.bounds = {k:[min(self.xinputs[k]), max(self.xinputs[k])] for k in self.xinputs_order()}
#         self.bounds['omnuh2'][0] = 0 


    def _get_region_dist(self,xinputs):
        xinput_vec = [xinputs[k] - self.xinput_region_center[k] for k in self.xinputs_order()]
        return sqrt(dot(xinput_vec,dot(self.xinput_region_icov,xinput_vec)))
    
    
    def check_inputs(self, inputs, xinputs=None):
        """Raise a CantUsePICO exception if these inputs aren't in the fit region."""
        if xinputs is None: xinputs = self.inputs_to_xinputs(inputs)
        for (k,(lower,upper)) in self.bounds.items():
            if not lower<=xinputs[k]<=upper: raise CantUsePICO("Parameter '%s'=%f is outside the PICO region bounds %f < %s < %f"%(k,xinputs[k],lower,k,upper))
        s = self._get_region_dist(xinputs)
        if s>1: 
            allinputs = xinputs.copy(); allinputs.update({k:v for k,v in inputs.items() if k in self.inputs()})
            raise CantUsePICO("Point is outside of PICO interpolation region. Distance from center is %.2f but needs to be less than 1. %s"%(s,allinputs))

    
    def dxinputs_to_dinputs(self,dxinputs,inputs):
        pass

    def get_garbage(self):
        """Attributes of this class which don't need to be stored in the data file."""
        return ['inputs','xinputs', 'coefficients','qr']

            
            
            

            
class DefaultOutputter(Cleanable):
    def __init__(self, inputs, outputs, xindices=None, indices=None):
        self.xindices = xindices
        self.indicies = indices
        self.xoutputs = self.outputs_to_xoutputs(outputs,inputs)
    
    def xoutputs_to_outputs(self,xoutputs,allinputs):
        if self.indicies!=None: 
            xoutputs = splev(self.indicies,splrep(self.xindices,xoutputs)) #ext=1
        return xoutputs
    
    def outputs_to_xoutputs(self,outputs,allinputs):
        return outputs
    
    def get_garbage(self):
        return ['xoutputs']


class SameOrderInputter(DefaultInputter):
    def __init__(self, order, *args, **kwargs):
        self.order = order
        super(SameOrderInputter,self).__init__(*args, **kwargs)
    
    def xinputs_order(self):
        inputs = ['Omega_k',
                 'A_s',
                 '100*theta_s',
                 'N_ur',
                 'omega_b',
                 'YHe',
                 'n_s',
                 'tau_reio',
                 'omega_cdm']
        return {i:self.order for i in inputs}

    
class ClppInputter(DefaultInputter):
    def xinputs_order(self):
        inputs = {'scalar_nrun(1)':3, 
                  'omnuh2':5, 
                  'omch2':5,
                  'theta':5, 
                  'massive_neutrinos':5, 
                  'scalar_spectral_index(1)':5, 
                  'ombh2':5, 
                  'scalar_amp(1)':6}
        return inputs


class MpkInputter(DefaultInputter):
    def xinputs_order(self):
        inputs = {'scalar_nrun(1)':3, 
                  'omnuh2':3, 
                  'omch2':3,
                  'theta':3, 
                  'massive_neutrinos':3, 
                  'scalar_spectral_index(1)':3, 
                  'ombh2':3, 
                  'scalar_amp(1)':3}
        return inputs

    
    
class ClppOutputter(DefaultOutputter):
    pass
    
    
class PicoFit(object):
    def __init__(self, inputter, outputter):
        self.inputter = inputter
        self.outputter = outputter
        q,r = inputter.qr
        self.x=solve(r,dot(q.T,outputter.xoutputs))

    def get_result(self,inputs,derive=None,force=False):
        xinputs = self.inputter.inputs_to_xinputs(**inputs)
        if not force: self.inputter.check_inputs(inputs, xinputs)
        coefficients = self.inputter.xinputs_to_coeffs(xinputs)
        allinputs = {}; allinputs.update(inputs); allinputs.update(xinputs);
        r = self.outputter.xoutputs_to_outputs(dot(transpose(coefficients),self.x),allinputs)
        return r