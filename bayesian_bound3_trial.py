from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from scipy.optimize import minimize
import emcee
import corner
from scipy import stats

mcrit = 1.1e14 ##1.1e17 g s^-1 or 1.1e14 kg/s

cs = 1. #In km/s

G = 6.67e-11

M_CMZ = 0.4*9.77e6  # In Msol 

phistar = 1e-13 #In erg cm^-2 s^-1
phistar_si = 1e-13*(1e-7)*(1e4)

r = 8.3*(3.08e19) #Choose distance from GC to earth be 8.3kpc
sph = 4*pi*(r**2)

## Define pdfs

def vel_dist(x, mu):
	a = (mu/2.)*sqrt(pi/2.)
	z1 = -(x**2)/(2*a**2)
	z2 = a**3
	return (sqrt(2/pi)/z2)*(x**2)*exp(z1)

def Pr(nmin, nmax, beta, A):
	a1 = A/(1.-beta)
	a2 = nmax**(1.-beta) - nmin**(1.-beta)
	return a1*a2  
	

def pdf_clouds(x, mu, nmin, nmax, beta, A):
	return vel_dist(x, mu)*Pr(nmin, nmax, beta, A)


#Define Nclouds - the total number of black holes

def Nclouds(M, fclouds):
	return fclouds*M_CMZ/M

##Define mcrit as a function of mass

def mcrit(M):
	return 1.26e31*M*0.01/(3e8*3e8*0.1)


## Define functions to obtain the parameter space

def n_min(phi_th, M, v, csin):
	Mkg = M*(1.989e30)
	num1 = phi_th*mcrit(M)*sph
	den1 = 0.3*0.1*20.079*(pi**2)*(G**4)*(Mkg**4)*(3e8)*(3e8)
	VD = (cs**2)/(2*csin)
	VR = 2*csin
	if v>=VR:
		rdel = sqrt(((v**2+cs**2)**2)-4*(v**2)*(csin**2))
		y_m = (v**2+cs**2-rdel)/(2*(csin**2))
		vterms = (v**2+(csin**2)*(y_m**2))**3
		vterms = vterms*1e18
		num = vterms*num1
		den = (y_m**8)*den1
		rhom_2 = num/den
		rho = (rhom_2)**0.5
	elif v<=VD:
		rdel = sqrt(((v**2+cs**2)**2)-4*(v**2)*(csin**2))
		y_p = (v**2+cs**2+rdel)/(2*(csin**2))
		vterms = (v**2+(csin**2)*(y_p**2))**3
		vterms = vterms*(1e18)
		num = vterms*num1
		den = (y_p**8)*den1
		rhop_2 = num/den
		rho = (rhop_2)**0.5
	else:
		y_z = (v**2+cs**2)/(2*(csin**2))
		vterms = 8*(csin**6)
		vterms = vterms*1e18
		num = vterms*num1
		den = den1*(y_z**2)
		rhoz_2 = num/den
		rho = (rhoz_2)**0.5
	con = 2.72*1.67e-27*1e6
	n_cgs = rho/con
	return n_cgs

def v_space(phi_th, M, csin):
	vbh = [i/10. for i in range(0,4001,1)]
	v_domain = []
	v_domain_excl = []
	for vl in vbh:
		if 200<=n_min(phi_th, M, vl, csin)<=1e6:
			v_domain.append(vl)
		elif n_min(phi_th, M, vl, csin)<200: 
			v_domain_excl.append(vl)
		else:
			pass
	return v_domain, v_domain_excl

def v_dfinder(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))
	
def p_sources(phi_th, mu_BH, csin, M, beta, A):
	v_d, v_d_excl = v_space(phi_th, M, csin)
	v_ds1 = v_dfinder(v_d)
	v_ds2 = v_dfinder(v_d_excl)
	Integral = 0.
	g1 = lambda v : pdf_clouds(v, mu_BH, n_min(phi_th, M, v, csin), 1e6, beta, A)
	g2 = lambda v : pdf_clouds(v, mu_BH, 200, 1e6, beta, A)
	if len(v_ds1)!=0:
		for i1 in range(len(v_ds1)):
			v_r1 = v_ds1[i1]
			vmin1 = v_r1[0]
			vmax1 = v_r1[1]
			p1 = integrate.quad(g1, vmin1, vmax1)
			Integral = Integral + p1[0]
	if len(v_ds2)!=0:
		for i2 in range(len(v_ds2)):
			v_r2 = v_ds2[i2]
			vmin2 = v_r2[0]
			vmax2 = v_r2[1]
			p2 = integrate.quad(g2, vmin2, vmax2)
			Integral = Integral + p2[0]
	return Integral


def N_s(phi_th, mu_BH, csin, M, beta, A, fclouds):
	prob = p_sources(phi_th, mu_BH, csin, M, beta, A)
	Ns = prob*Nclouds(M, fclouds)
	return Ns

Nobs = 70

def fPBH_solve(phi_th, mu_BH, csin, M, beta, A, fclouds):
	if N_s(phi_th, mu_BH, csin, M, beta, A, fclouds)>Nobs:
		f_pbh = Nobs/N_s(phi_th, mu_BH, csin, M, beta, A, fclouds)
	else:
		f_pbh = 1.
	return f_pbh


##Define a constant log_prior

def log_prior(theta):
	fPBH, mu_BH, csin = theta	
	lp = 0.
	if 0.<=fPBH<=1. and 0.1<=mu_BH<=250. and 1.<=csin<=45.:
		lp = 0.
	else:
		lp = -np.inf
	return lp


def Log_likelihood(phi_th, mu_BH, csin, M, beta, A, fclouds, fPBH):
	N_mu_f = N_s(phi_th, mu_BH, csin, M, beta, A, fclouds)*fPBH
	numt1 = (N_mu_f**Nobs)*exp(-N_mu_f)
	dent1 = factorial(Nobs)
	if numt1>0. and dent1>0:
		ll1 = log(numt1) - log(dent1)
	else:
		ll1 = -np.inf
	return ll1


##Define the log likelihood function fixing other parameters
def log_likelihood2(theta):
	fPBH, mu_BH, csin = theta
	return Log_likelihood(phistar_si, mu_BH, csin, 40., 2.4, 2333.33, 0.2214, fPBH)	


## Minimize the above loglikelihood function 	
np.random.seed(81)
nll1 = lambda *args: -log_likelihood2(*args)


fPBH_initial = 0.005
mu_BH_initial = 145.
csin_initial = 25.

soln = minimize(nll1, [fPBH_initial, mu_BH_initial, csin_initial])	

f_ml = soln.x ##fPBH obtained from maximum likelihood estimation

## define log probability

def log_probability(theta):
	fPBH, mu_BH, csin = theta
	lpr = log_prior(theta)
	if not np.isfinite(lpr):
		return -np.inf
	return lpr + log_likelihood2(theta)


#Define position around ml estimate

pos = soln.x + 1e-5*np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 15000, progress=True)

labels = ["fPBH", "mu_BH", "csin"]
samples = sampler.get_chain()

for idim in range(ndim):
	plt.figure(idim)
	plt.plot(samples[:,:,idim], color='k')
	plt.ylabel(labels[idim])
	plt.xlabel("Number of steps")
	plt.title("Plot of "+labels[idim]+" as a function of number of steps, for 40 M_sol black hole")
	plt.savefig("bayesian_bound3_15000_"+labels[idim]+"_40Msol.png")
	#plt.show()
	
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

##Store the data of histogram

fn1 = open('/home/tejas/bayesian3_data_15000_40Msol.txt', 'w')
for i in range(len(flat_samples)):
	print(flat_samples[i][0], flat_samples[i][1], flat_samples[i][2], file = fn1)

fn1.close()

fig = corner.corner(flat_samples, labels=labels, truths=f_ml)
fig.savefig("bayesian_posterior_multi_40Msol_15000.png")

ftau = open("/home/tejas/bayesian3_tau_trial_15000_40Msol.txt", "w")

tau = sampler.get_autocorr_time()
print(tau, file=ftau)
ftau.close()
