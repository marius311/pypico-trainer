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

    import cosmoslik as K
    from pandas import DataFrame
    
    ntest = 500
    
    lvals = loadtxt("lvalues",dtype=int)
    lvals = lvals[lvals<2510]
    
    names,ps,cl = pickle.load(open("results.dat"))
    idxs = array([len(x)==5500 for x in cl])
    cl = vstack(cl[idxs])[:-ntest]
    ps = ps[idxs][:-ntest]
    
    name_mapping = {
        'hubble':'H0',
        'omch2':'omch2',
        'ombh2':'ombh2',
        'omk':'omk',
        're_optical_depth':'tau',
        'scalar_amp(1)':'As',
        'scalar_spectral_index(1)':'ns'
    }
    names = [name_mapping[n] for n in names]
    
    
    inputs = dict(zip(names,ps.T))
    inputter = SameOrderInputter(4,inputs=inputs)

    outputs = cl[:,lvals]
    outputter = DefaultOutputter(inputs,outputs,xindices=lvals,indices=arange(max(lvals)))
    
    fit = PicoFit(inputter,outputter)
    
    pico = jcdemo(fit)

    return pico


def fixCls(TT,EE,TE):
    """Make auto-spectra positive and correlation matrix positive definite."""
    TT[TT<0]=0; EE[EE<0]=0
    r=TE/sqrt(TT*EE)
    r[r>1]=1; r[r<-1]=-1
    return TT,EE,r*sqrt(TT*EE)


class jcdemo(PICO):

    def __init__(self,fit):
        self._fit = fit


    def get(self, derive=None, force=False, **inputs):
        """
        Run PICO to get some outputs given some set of inputs.

        Parameters
        ----------

        **inputs : A dictionary of the values of the inputs. Valid input names
                   are given by PICO.inputs() and an example set of inputs
                   is given by PICO.example_inputs()

        outputs, optional : A list of outputs which to actually get. You can speed up
                            computation time by only getting those outputs you need.
                            Valid outputs are given by PICO.outputs().
                            (default: all of them)

        derive, optional : Compute dCl/dparam. `derive` should be a dictionary
                           giving the number of derivatives with respect to each parameter.

        force, optional : Use PICO even if the point is outside the training region
                          and may return an innaccurate result.


        """
        result = {}

        #Convert pivots
        pivot_ratio = 0.05 / inputs.get('pivot_scalar',0.05)
        inputs['As'] = inputs['As']*pivot_ratio**(inputs['ns']-1)

        #Compute results
        result = {'dl_TT': self._fit.get_result(inputs,derive=derive,force=force)}

        return result


    def inputs(self):
        return self._fit.inputter.inputs_used()

    def outputs(self):
        return ['cl_TT']

    def example_inputs(self):
        """
        An example set of inputs, which can be passed to pico.get(**pico.example_inputs())
        """
        return {'As': 2.08e-9,
                'ns': 0.97,
                'tau': 0.055,
                'ombh2': 0.0225,
                'omch2': 0.117,
                'H0': 50,
                'omk': -0.05,
                'pivot_scalar':0.05}


class Cleanupable(object):

    def get_garbage():
        """Attributes from this class to be deleted."""
        return []

    def clean_up(self):
        """Get rid of things which don't need to be stored in the datafile. """
        for a in self.get_garbage():
            try: delattr(self,a)
            except AttributeError: pass


class DefaultInputter(Cleanupable):
    """
    (input -> xinput)
    """
    def __init__(self, inputs):
        self.inputs = inputs
        self.xinputs = self.inputs_to_xinputs(**inputs)
        self.xinput_center = self.get_xinput_center(self.xinputs)
        self.poly = getpoly(self.xinputs_order().keys(),self.xinputs_order())
        self.coefficients = self.xinputs_to_coeffs(self.xinputs)
        self.qr = qr(self.coefficients)
        self.input_center = self.get_xinput_center(self.xinputs)
        self.calc_bounds()


    def inputs_used(self):
        return ['As',
                'ns',
                'tau',
                'ombh2',
                'omch2',
                'H0',
                'omk']

    def inputs_to_xinputs(self,**inputs):
        inputs['ommh3'] = (inputs['omch2'] + inputs['ombh2'])*(inputs['H0']/100.)
        return inputs

    def get_xinput_center(self,xinputs):
        return {k:mean(v) for k,v in xinputs.items()}

    def xinputs_to_coeffs(self,xinputs,derive=None):
        """Given a set of xinputs, get the terms of the fitting polynomial."""
        return array(self.poly({k:(xinputs[k]-self.xinput_center[k]) for k in self.xinputs_order().keys()},derive=derive)).T

    def calc_bounds(self):
        self.xinput_region_center = {k:mean(self.xinputs[k]) for k in self.xinputs_order()}
        xinput_matrix = vstack([self.xinputs[k] - self.xinput_region_center[k] for k in self.xinputs_order()])
        icov = inv(cov(xinput_matrix))
        self.xinput_region_icov = icov / max(sqrt(sum(xinput_matrix*dot(icov,xinput_matrix),axis=0)))**2
        self.bounds = {k:[min(self.xinputs[k]), max(self.xinputs[k])] for k in self.xinputs_order()}


    def _get_region_dist(self,xinputs):
        xinput_vec = [xinputs[k] - self.xinput_region_center[k] for k in self.xinputs_order()]
        return sqrt(dot(xinput_vec,dot(self.xinput_region_icov,xinput_vec)))


    def check_inputs(self, inputs, xinputs=None):
        """Raise a CantUsePICO exception if these inputs aren't in the fit region."""
        if xinputs is None: xinputs = self.inputs_to_xinputs(inputs)
        for (k,(lower,upper)) in self.bounds.items():
            if not lower<=xinputs[k]<=upper: raise CantUsePICO("Parameter '%s'=%.5g is outside the PICO region bounds %.5g < %s < %.5g"%(k,xinputs[k],lower,k,upper))
        s = self._get_region_dist(xinputs)
        if s>1:
            allinputs = xinputs.copy(); allinputs.update({k:v for k,v in inputs.items() if k in self.inputs_used()})
            raise CantUsePICO("Point is outside of PICO interpolation region. Distance from center is %.2f but needs to be less than 1. %s"%(s,allinputs))


    def dxinputs_to_dinputs(self,dxinputs,inputs):
        pass

    def get_garbage(self):
        """Attributes of this class which don't need to be stored in the data file."""
        return ['inputs','xinputs','qr','coefficients']




class DefaultOutputter(Cleanupable):
    def __init__(self, inputs, outputs, xindices=None, indices=None):
        self.xindices = xindices
        self.indicies = indices
        self.xoutputs = self.outputs_to_xoutputs(outputs,inputs)

    def xoutputs_to_outputs(self,xoutputs,allinputs):
        if self.indicies is not None:
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
        inputs = ['As',
                  'ns',
                  'tau',
                  'ombh2',
                  'ommh3',
                  'H0',
                  'omk']
        return {i:self.order for i in inputs}

class PicoFit(object):
    def __init__(self, inputter, outputter):
        self.inputter = inputter
        self.outputter = outputter
        q,r = inputter.qr
        self.x=solve(r,dot(q.T,outputter.xoutputs))

    def get_result(self,inputs,derive=None,force=False):
        xinputs = self.inputter.inputs_to_xinputs(**inputs)
        if not force: self.inputter.check_inputs(inputs, xinputs)
        coefficients = self.inputter.xinputs_to_coeffs(xinputs,derive=derive)
        allinputs = {}; allinputs.update(inputs); allinputs.update(xinputs);
        r = self.outputter.xoutputs_to_outputs(dot(transpose(coefficients),self.x),allinputs)
        return r
