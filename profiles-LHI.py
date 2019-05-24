# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:46:24 2019

@author: mathewsa
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Gibbs_kern, ConstantKernel as C
import seaborn as sns
import matplotlib
sns.set_style("whitegrid",{'axes.facecolor': 'white','axes.grid': False,}) 
import matplotlib.style as style 
style.available 
#style.use('seaborn-poster') #sets the size of the charts
#style.use('ggplot')
sns.set_context("poster")  
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context('talk')
matplotlib.rcParams['font.family'] = "serif"

def noise(x):
    noise_out = 0.3*(1. - np.tanh(100.*(x-0.03))) + 0.05*x*(x-0.02) + 0.05 + 0.05*(1.-x)* np.random.random(x.shape)
    return noise_out

with open('/Users/mathewsa/Downloads/L_mode_T.csv', newline='') as csvfile:
    L_mode_T = list(csv.reader(csvfile))
L_mode_T_x = np.array([float(item[0]) for item in L_mode_T if float(item[0]) > -0.04])
L_mode_T_y = [float(item[1]) for item in L_mode_T if float(item[0]) > -0.04]
with open('/Users/mathewsa/Downloads/H_mode_T.csv', newline='') as csvfile:
    H_mode_T = list(csv.reader(csvfile))
H_mode_T_x = np.array([float(item[0]) for item in H_mode_T if float(item[0]) > -0.04])
H_mode_T_y = [float(item[1]) for item in H_mode_T if float(item[0]) > -0.04]
with open('/Users/mathewsa/Downloads/I_mode_T.csv', newline='') as csvfile:
    I_mode_T = list(csv.reader(csvfile))
I_mode_T_x = np.array([float(item[0]) for item in I_mode_T if float(item[0]) > -0.04])
I_mode_T_y = [float(item[1]) for item in I_mode_T if float(item[0]) > -0.04]

with open('/Users/mathewsa/Downloads/L_mode_n.csv', newline='') as csvfile:
    L_mode_density = list(csv.reader(csvfile))
L_mode_density_x = np.array([float(item[0]) for item in L_mode_density if float(item[0]) > -0.04])
L_mode_density_y = [float(item[1]) for item in L_mode_density if float(item[0]) > -0.04]
with open('/Users/mathewsa/Downloads/H_mode_n.csv', newline='') as csvfile:
    H_mode_density = list(csv.reader(csvfile))
H_mode_density_x = np.array([float(item[0]) for item in H_mode_density if float(item[0]) > -0.04])
H_mode_density_y = [float(item[1]) for item in H_mode_density if float(item[0]) > -0.04]
with open('/Users/mathewsa/Downloads/I_mode_n.csv', newline='') as csvfile:
    I_mode_density = list(csv.reader(csvfile))
I_mode_density_x = np.array([float(item[0]) for item in I_mode_density if float(item[0]) > -0.04])
I_mode_density_y = [float(item[1]) for item in I_mode_density if float(item[0]) > -0.04] 

with open('/Users/mathewsa/Downloads/L-mode_T.csv', newline='') as csvfile:
    L_mode_T = list(csv.reader(csvfile))
L_mode_T_x1 = np.array([float(item[0]) for item in L_mode_T])
L_mode_T_y1 = [float(item[1]) for item in L_mode_T]
with open('/Users/mathewsa/Downloads/H-mode_T.csv', newline='') as csvfile:
    H_mode_T = list(csv.reader(csvfile))
H_mode_T_x1 = np.array([float(item[0]) for item in H_mode_T])
H_mode_T_y1 = [float(item[1]) for item in H_mode_T]
with open('/Users/mathewsa/Downloads/I-mode_T.csv', newline='') as csvfile:
    I_mode_T = list(csv.reader(csvfile))
I_mode_T_x1 = np.array([float(item[0]) for item in I_mode_T])
I_mode_T_y1 = [float(item[1]) for item in I_mode_T]

with open('/Users/mathewsa/Downloads/L-mode-density.csv', newline='') as csvfile:
    L_mode_density = list(csv.reader(csvfile))
L_mode_density_x1 = np.array([float(item[0]) for item in L_mode_density])
L_mode_density_y1 = [float(item[1]) for item in L_mode_density]
with open('/Users/mathewsa/Downloads/H-mode-density.csv', newline='') as csvfile:
    H_mode_density = list(csv.reader(csvfile))
H_mode_density_x1 = np.array([float(item[0]) for item in H_mode_density])
H_mode_density_y1 = [float(item[1]) for item in H_mode_density]
with open('/Users/mathewsa/Downloads/I-mode-density.csv', newline='') as csvfile:
    I_mode_density = list(csv.reader(csvfile))
I_mode_density_x1 = np.array([float(item[0]) for item in I_mode_density])
I_mode_density_y1 = [float(item[1]) for item in I_mode_density]

plt.figure()
plt.errorbar(L_mode_T_x, L_mode_T_y,np.zeros(len(L_mode_T_x)), fmt='k.', markersize=15, label=u'L-mode',alpha=0.3)
plt.errorbar(H_mode_T_x, H_mode_T_y,np.zeros(len(H_mode_T_x)), fmt='g.', markersize=15, label=u'H-mode',alpha=0.3)
plt.errorbar(I_mode_T_x, I_mode_T_y,np.zeros(len(I_mode_T_x)), fmt='r.', markersize=15, label=u'I-mode',alpha=0.3)
plt.plot(L_mode_T_x1, L_mode_T_y1, 'k--',alpha=0.3)
plt.plot(H_mode_T_x1, H_mode_T_y1, 'g--',alpha=0.3)
plt.plot(I_mode_T_x1, I_mode_T_y1, 'r--',alpha=0.3)
#plt.errorbar(X.ravel(), y, noise, fmt='g.', markersize=10, label=u'Observations')
#plt.plot(x, y_pred, 'r-', label=u'Prediction')
#plt.fill(np.concatenate([x, x[::-1]]),
#         np.concatenate([y_pred - 1.9600 * sigma,
#                        (y_pred + 1.9600 * sigma)[::-1]]),
#         alpha=.1, fc='r', ec='None', label='95% confidence interval')
#plt.xlabel(r"$\rho$")
#plt.ylabel(r"$n_e \ (10^{20} \ $"+"m"+r"$^{-3})$")
plt.xlabel("R - R"+r"$_{LCFS} \ $"+"(m)")
plt.ylabel("T"+r"$_e \ $"+"(eV)")
plt.axvline(x=0.0, color='k', linestyle='--') 
plt.xlim(-0.04, 0.006)
plt.ylim(0, 1600)
plt.axvline(x=0.0, color='k', linestyle='--') 
plt.axvline(x=-0.015, color='k', linestyle='--')
plt.text(0.003, 1450, 'SOL', size=16, horizontalalignment='center', verticalalignment='center')
plt.text(-0.0076, 1450, 'PEDESTAL', size=16, horizontalalignment='center', verticalalignment='center')
plt.text(-0.028, 1450, 'CORE PLASMA', size=16, horizontalalignment='center', verticalalignment='center')
plt.text(-0.001, 1000, 'LCFS', size=16, rotation = 90, horizontalalignment='center', verticalalignment='center')
bbox_props = dict(boxstyle="rarrow", fc=(0.9, 0.9, 0.9), ec="k", lw=4, alpha=0.3)
t = plt.text(-0.028, 1275., "         ", ha="center", va="center", rotation=180,
            size=10,
            bbox=bbox_props)
plt.legend(loc='lower left',fontsize=12) 
plt.show()

plt.figure()
plt.errorbar(L_mode_density_x, L_mode_density_y,np.zeros(len(L_mode_density_x)), fmt='k.', markersize=15, label=u'L-mode',alpha=0.3)
plt.errorbar(H_mode_density_x, H_mode_density_y,np.zeros(len(H_mode_density_x)), fmt='g.', markersize=15, label=u'H-mode',alpha=0.3)
plt.errorbar(I_mode_density_x, I_mode_density_y,np.zeros(len(I_mode_density_x)), fmt='r.', markersize=15, label=u'I-mode',alpha=0.3)
plt.plot(L_mode_density_x1, L_mode_density_y1, 'k--',alpha=0.3)
plt.plot(H_mode_density_x1, H_mode_density_y1, 'g--',alpha=0.3)
plt.plot(I_mode_density_x1, I_mode_density_y1, 'r--',alpha=0.3)
#plt.errorbar(X.ravel(), y, noise, fmt='g.', markersize=10, label=u'Observations')
#plt.plot(x, y_pred, 'r-', label=u'Prediction')
#plt.fill(np.concatenate([x, x[::-1]]),
#         np.concatenate([y_pred - 1.9600 * sigma,
#                        (y_pred + 1.9600 * sigma)[::-1]]),
#         alpha=.1, fc='r', ec='None', label='95% confidence interval')
#plt.xlabel(r"$\rho$")
plt.ylabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$")
plt.xlabel("R - R"+r"$_{LCFS} \ $"+"(m)") 
#plt.axvline(x=0.0, color='k', linestyle='--') 
#plt.axvline(x=-0.015, color='k', linestyle='--')
#plt.text(0.003, 2.25, 'SOL', size=16, horizontalalignment='center', verticalalignment='center')
#plt.text(-0.0076, 2.25, 'PEDESTAL', size=16, horizontalalignment='center', verticalalignment='center')
#plt.text(-0.028, 2.25, 'CORE PLASMA', size=16, horizontalalignment='center', verticalalignment='center')
#plt.text(-0.001, 1.55, 'LCFS', size=16, rotation = 90, horizontalalignment='center', verticalalignment='center')
plt.xlim(-0.04, 0.006)
plt.ylim(0, 2.5)
plt.text(-0.003, 2.25, 'SHOT #1091016033', size=10, horizontalalignment='center', verticalalignment='center')
#bbox_props = dict(boxstyle="rarrow", fc=(0.9, 0.9, 0.9), ec="k", lw=4, alpha=0.3)
#t = plt.text(-0.028, 1.95, "         ", ha="center", va="center", rotation=180,
#            size=10,
#            bbox=bbox_props)
plt.legend(loc='lower left',fontsize=12) 
plt.show()


#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#
#ax1.errorbar(L_mode_T_x, L_mode_T_y,np.zeros(len(L_mode_T_x)), fmt='b.', markersize=15, label=u'L-mode',alpha=0.5)
#ax1.errorbar(H_mode_T_x, H_mode_T_y,np.zeros(len(H_mode_T_x)), fmt='r.', markersize=15, label=u'H-mode',alpha=0.5)
#ax1.errorbar(I_mode_T_x, I_mode_T_y,np.zeros(len(I_mode_T_x)), fmt='g.', markersize=15, label=u'I-mode',alpha=0.5)
#ax1.plot(L_mode_T_x1, L_mode_T_y1, 'b--')
#ax1.plot(H_mode_T_x1, H_mode_T_y1, 'r--')
#ax1.plot(I_mode_T_x1, I_mode_T_y1, 'g--')
#ax1.set_xlabel(r"$R - R_{LCFS} \ $"+"(m)")
#ax1.set_ylabel(r"$T_e \ $"+"(eV)")
#ax1.set_xlim(-0.04, 0.01)
#ax1.set_ylim(0, 2000)
#
#ax2.errorbar(L_mode_density_x, L_mode_density_y,np.zeros(len(L_mode_density_x)), fmt='b.', markersize=15, label=u'L-mode',alpha=0.5)
#ax2.errorbar(H_mode_density_x, H_mode_density_y,np.zeros(len(H_mode_density_x)), fmt='r.', markersize=15, label=u'H-mode',alpha=0.5)
#ax2.errorbar(I_mode_density_x, I_mode_density_y,np.zeros(len(I_mode_density_x)), fmt='g.', markersize=15, label=u'I-mode',alpha=0.5)
#ax2.plot(L_mode_density_x1, L_mode_density_y1, 'b--')
#ax2.plot(H_mode_density_x1, H_mode_density_y1, 'r--')
#ax2.plot(I_mode_density_x1, I_mode_density_y1, 'g--')
#ax2.set_ylabel(r"$n_e \ (10^{20} \ $"+"m"+r"$^{-3})$") 
#ax2.set_ylim(0, 2.75)
#  
#plt.legend(loc='upper right') 
#plt.show() 

#from scipy.optimize import curve_fit
#def func(x, b, h, R0, d, m):
#    return b + (h/2.)*(np.tanh((R0-x)/d) + 1.) + m*(R0 - x - d)*np.heaviside((R0 - x - d),x)
#
#plt.figure()
#plt.errorbar(L_mode_density_x, L_mode_density_y,np.zeros(len(L_mode_density_x)), fmt='b.', markersize=15, label=u'L-mode',alpha=0.5)
#plt.errorbar(H_mode_density_x, H_mode_density_y,np.zeros(len(H_mode_density_x)), fmt='r.', markersize=15, label=u'H-mode',alpha=0.5)
#plt.errorbar(I_mode_density_x, I_mode_density_y,np.zeros(len(I_mode_density_x)), fmt='g.', markersize=15, label=u'I-mode',alpha=0.5)
##>>> xdata = np.linspace(0, 4, 50)
##>>> y = func(xdata, 2.5, 1.3, 0.5)
##>>> np.random.seed(1729)
##>>> y_noise = 0.2 * np.random.normal(size=xdata.size)
##>>> ydata = y + y_noise
##>>> plt.plot(xdata, ydata, 'b-', label='data')
#x_T = np.linspace(-0.075, 0.01, 1000)
#x_n = np.linspace(-0.075, 0.01, 1000)
#popt_L_mode_density, pcov = curve_fit(func, L_mode_density_x, L_mode_density_y,maxfev=100000)
#plt.plot(x_n, func(x_n, *popt_L_mode_density), 'b-')
#popt_H_mode_density, pcov = curve_fit(func, H_mode_density_x, H_mode_density_y,maxfev=100000)
#plt.plot(x_n, func(x_n, *popt_H_mode_density), 'r-')
#popt_I_mode_density, pcov = curve_fit(func, I_mode_density_x, I_mode_density_y,maxfev=100000)
#plt.plot(x_n, func(x_n, *popt_I_mode_density), 'g-')
#plt.ylabel(r"$n_e \ (10^{20} \ $"+"m"+r"$^{-3})$")
#plt.xlabel(r"$R - R_{LCFS} \ $"+"(m)") 
#plt.xlim(-0.04, 0.01)
#plt.ylim(0, 2.75)
#plt.legend(loc='upper right') 
#plt.show()


#plt.figure()
#plt.errorbar(L_mode_T_x, L_mode_T_y,np.zeros(len(L_mode_T_x)), fmt='b.', markersize=15, label=u'L-mode',alpha=0.5)
#plt.errorbar(H_mode_T_x, H_mode_T_y,np.zeros(len(H_mode_T_x)), fmt='r.', markersize=15, label=u'H-mode',alpha=0.5)
#plt.errorbar(I_mode_T_x, I_mode_T_y,np.zeros(len(I_mode_T_x)), fmt='g.', markersize=15, label=u'I-mode',alpha=0.5)
#popt_L_mode_T, pcov = curve_fit(func, L_mode_T_x, L_mode_T_y,maxfev=1000000)
#plt.plot(x_T, func(x_T, *popt_L_mode_T), 'b-')
#popt_H_mode_T, pcov = curve_fit(func, H_mode_T_x, H_mode_T_y,maxfev=1000000)
#plt.plot(x_T, func(x_T, *popt_H_mode_T), 'r-')
#popt_I_mode_T, pcov = curve_fit(func, I_mode_T_x, I_mode_T_y,maxfev=1000000)
#plt.plot(x_T, func(x_T, *popt_I_mode_T), 'g-') 
#plt.xlabel(r"$R - R_{LCFS} \ $"+"(m)") 
#plt.ylabel(r"$T_e \ $"+"(eV)")
#plt.xlim(-0.04, 0.01)
#plt.ylim(0, 2000)
#plt.legend(loc='upper right') 
#plt.show()

x_T = np.atleast_2d(np.linspace(-0.07, 0.01, 1000)).T
x_n = np.atleast_2d(np.linspace(-0.075, 0.01, 1000)).T
kernel_T = C(10.0, (1e-5, 1e5)) * RBF(10.0, (1e-5, 1e5))
kernel_n = C(0.01, (1e-5, 1e-1)) * RBF(0.01, (1e-5, 1e5))

gp_L_mode_T = GaussianProcessRegressor(kernel=kernel_T, alpha=(200.*noise(L_mode_T_x))**2.,
                              n_restarts_optimizer=10)
L_mode_T_X = np.atleast_2d(L_mode_T_x).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp_L_mode_T.fit(L_mode_T_X, L_mode_T_y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
L_mode_T_yp, L_mode_T_sigma = gp_L_mode_T.predict(x_T, return_std=True)

gp_H_mode_T = GaussianProcessRegressor(kernel=kernel_T, alpha=(200.*noise(H_mode_T_x))**2.,
                              n_restarts_optimizer=10)
H_mode_T_X = np.atleast_2d(H_mode_T_x).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp_H_mode_T.fit(H_mode_T_X, H_mode_T_y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
H_mode_T_yp, H_mode_T_sigma = gp_H_mode_T.predict(x_T, return_std=True)

gp_I_mode_T = GaussianProcessRegressor(kernel=kernel_T, alpha=(200.*noise(I_mode_T_x))**2.,
                              n_restarts_optimizer=10)
I_mode_T_X = np.atleast_2d(I_mode_T_x).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp_I_mode_T.fit(I_mode_T_X, I_mode_T_y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
I_mode_T_yp, I_mode_T_sigma = gp_I_mode_T.predict(x_T, return_std=True)


gp_L_mode_density = GaussianProcessRegressor(kernel=kernel_n, alpha=(0.3*noise(L_mode_density_x))**2.,
                              n_restarts_optimizer=10)
L_mode_density_X = np.atleast_2d(L_mode_density_x).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp_L_mode_density.fit(L_mode_density_X, L_mode_density_y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
L_mode_density_yp, L_mode_density_sigma = gp_L_mode_density.predict(x_n, return_std=True)

gp_H_mode_density = GaussianProcessRegressor(kernel=kernel_n, alpha=(0.3*noise(H_mode_density_x))**2.,
                              n_restarts_optimizer=10)
H_mode_density_X = np.atleast_2d(H_mode_density_x).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp_H_mode_density.fit(H_mode_density_X, H_mode_density_y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
H_mode_density_yp, H_mode_density_sigma = gp_H_mode_density.predict(x_n, return_std=True)

gp_I_mode_density = GaussianProcessRegressor(kernel=kernel_n, alpha=(0.3*noise(I_mode_density_x))**2.,
                              n_restarts_optimizer=10)
I_mode_density_X = np.atleast_2d(I_mode_density_x).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp_I_mode_density.fit(I_mode_density_X, I_mode_density_y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
I_mode_density_yp, I_mode_density_sigma = gp_I_mode_density.predict(x_n, return_std=True)


plt.figure()
plt.errorbar(L_mode_T_x, L_mode_T_y,200.*noise(L_mode_T_x), fmt='b.', markersize=15, label=u'L-mode',alpha=0.5)
plt.errorbar(H_mode_T_x, H_mode_T_y,200.*noise(H_mode_T_x), fmt='r.', markersize=15, label=u'H-mode',alpha=0.5)
plt.errorbar(I_mode_T_x, I_mode_T_y,200.*noise(I_mode_T_x), fmt='g.', markersize=15, label=u'I-mode',alpha=0.5)
plt.plot(x_T, L_mode_T_yp, 'b-')
plt.fill(np.concatenate([x_T, x_T[::-1]]),
         np.concatenate([L_mode_T_yp - 1.9600 * L_mode_T_sigma,
                        (L_mode_T_yp + 1.9600 * L_mode_T_sigma)[::-1]]),
         alpha=.1, fc='b', ec='None', label='95% confidence interval')
plt.plot(x_T, H_mode_T_yp, 'r-')
plt.fill(np.concatenate([x_T, x_T[::-1]]),
         np.concatenate([H_mode_T_yp - 1.9600 * H_mode_T_sigma,
                        (H_mode_T_yp + 1.9600 * H_mode_T_sigma)[::-1]]),
         alpha=.1, fc='r', ec='None', label='95% confidence interval')
plt.plot(x_T, I_mode_T_yp, 'g-')
plt.fill(np.concatenate([x_T, x_T[::-1]]),
         np.concatenate([I_mode_T_yp - 1.9600 * I_mode_T_sigma,
                        (I_mode_T_yp + 1.9600 * I_mode_T_sigma)[::-1]]),
         alpha=.1, fc='g', ec='None', label='95% confidence interval')
#plt.xlabel(r"$\rho$")
#plt.ylabel(r"$n_e \ (10^{20} \ $"+"m"+r"$^{-3})$")
plt.xlabel(r"$R - R_{LCFS} \ $"+"(m)")
plt.ylabel(r"$T_e \ $"+"(eV)")
plt.xlim(-0.04, 0.01)
plt.ylim(0, 2000)
plt.legend(loc='upper right') 
plt.show()

plt.figure()
plt.errorbar(L_mode_density_x, L_mode_density_y,0.3*noise(L_mode_density_x), fmt='b.', markersize=15, label=u'L-mode',alpha=0.5)
plt.errorbar(H_mode_density_x, H_mode_density_y,0.3*noise(H_mode_density_x), fmt='r.', markersize=15, label=u'H-mode',alpha=0.5)
plt.errorbar(I_mode_density_x, I_mode_density_y,0.3*noise(I_mode_density_x), fmt='g.', markersize=15, label=u'I-mode',alpha=0.5)
plt.plot(x_n, L_mode_density_yp, 'b-')
plt.fill(np.concatenate([x_n, x_n[::-1]]),
         np.concatenate([L_mode_density_yp - 1.9600 * L_mode_density_sigma,
                        (L_mode_density_yp + 1.9600 * L_mode_density_sigma)[::-1]]),
         alpha=.1, fc='b', ec='None', label='95% confidence interval')
plt.plot(x_n, H_mode_density_yp, 'r-')
plt.fill(np.concatenate([x_n, x_n[::-1]]),
         np.concatenate([H_mode_density_yp - 1.9600 * H_mode_density_sigma,
                        (H_mode_density_yp + 1.9600 * H_mode_density_sigma)[::-1]]),
         alpha=.1, fc='r', ec='None', label='95% confidence interval')
plt.plot(x_n, I_mode_density_yp, 'g-')
plt.fill(np.concatenate([x_n, x_n[::-1]]),
         np.concatenate([I_mode_density_yp - 1.9600 * I_mode_density_sigma,
                        (I_mode_density_yp + 1.9600 * I_mode_density_sigma)[::-1]]),
         alpha=.1, fc='g', ec='None', label='95% confidence interval')
#plt.xlabel(r"$\rho$")
#plt.ylabel(r"$n_e \ (10^{20} \ $"+"m"+r"$^{-3})$")
plt.xlabel(r"$R - R_{LCFS} \ $"+"(m)")
plt.ylabel(r"$n_e \ (10^{20} \ $"+"m"+r"$^{-3})$") 
plt.xlim(-0.04, 0.01)
plt.ylim(0, 2.75)
plt.legend(loc='upper right') 
plt.show()


#kernel_new = C(0.01, (1e-5, 1e-1)) * Gibbs_kern([0.1,1.,1.,1.],[(1e-2, 5e2),(1e-2, 1e3),(1e-2, 1e4),(1e-2, 1e5)])
#
#gp_L_mode_new = GaussianProcessRegressor(kernel=kernel_new, alpha=(200.*noise(L_mode_T_x))**2.,
#                              n_restarts_optimizer=10)
#gp_L_mode_new.fit(L_mode_density_X, L_mode_density_y)
#L_mode_density_yp_new, L_mode_density_sigma_new = gp_L_mode_new.predict(x_n, return_std=True)
##plt.figure()
##plt.plot(x_T, L_mode_T_yp, 'b-')
##plt.fill(np.concatenate([x_T, x_T[::-1]]),
##         np.concatenate([L_mode_T_yp - 1.9600 * L_mode_T_sigma,
##                        (L_mode_T_yp + 1.9600 * L_mode_T_sigma)[::-1]]),
##         alpha=.1, fc='b', ec='None', label='95% confidence interval')

