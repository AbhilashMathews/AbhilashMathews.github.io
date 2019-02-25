# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:33:29 2019

@author: mathewsa
"""

import numpy as np
from matplotlib import pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import stationary_sigmoid, Matern
from mpl_toolkits.mplot3d import Axes3D

from scipy.linalg import cholesky, cho_solve, solve_triangular

import sys
sys.path.append('C:/Users/mathewsa/') #provides path to gp_extras
from gp_extras.kernels import ManifoldKernel
from gp_extras.kernels import HeteroscedasticKernel

from sklearn.cluster import KMeans

#  Example independent variable (observations)
X = np.array([[0.,0.], [1.,0.], [2.,0.], [3.,0.], [4.,0.], 
                [5.,0.], [6.,0.], [7.,0.], [8.,0.], [9.,0.], [10.,0.], 
                [11.,0.], [12.,0.], [13.,0.], [14.,0.],
                [0.,1.], [1.,1.], [2.,1.], [3.,1.], [4.,1.], 
                [5.,1.], [6.,1.], [7.,1.], [8.,1.], [9.,1.], [10.,1.], 
                [11.,1.], [12.,1.], [13.,1.], [14.,1.],
                [0.,2.], [1.,2.], [2.,2.], [3.,2.], [4.,2.], 
                [5.,2.], [6.,2.], [7.,2.], [8.,2.], [9.,2.], [10.,2.], 
                [11.,2.], [12.,2.], [13.,2.], [14.,2.]])#.T

# Example dependent variable (observations) - noiseless case 
y = np.array([4.0, 3.98, 4.01, 3.95, 3.9, 3.84,3.8,
              3.73, 2.7, 1.64, 0.62, 0.59, 0.3, 
              0.1, 0.1,
            4.4, 3.9, 4.05, 3.9, 3.5, 3.4,3.3,
              3.23, 2.6, 1.6, 0.6, 0.5, 0.32, 
              0.05, 0.02,
            4.0, 3.86, 3.88, 3.76, 3.6, 3.4,3.2,
              3.13, 2.5, 1.6, 0.55, 0.51, 0.23, 
              0.11, 0.01]) 

#X = np.array([[0.,0.], [1.,0.], [2.,0.],
#                [0.,1.], [1.,1.], [2.,1.],
#                [0.,2.], [1.,2.], [2.,2.]])#.T
#
## Example dependent variable (observations) - noiseless case 
#y = np.array([4.0, 3.98, 4.01,
#            4.4, 3.9, 4.05,
#            4.0, 3.86, 3.88]) 

len_x1 = 20
len_x2 = 100
x1_min = 0
x2_min = 0
x1_max = 14
x2_max = 5
x1 = np.linspace(x1_min, x1_max, len_x1)
x2 = np.linspace(x2_min, x2_max, len_x2) 

i = 0 
inputs_x = []
while i < len(x1):
    j = 0
    while j < len(x2):
        inputs_x.append([x1[i],x2[j]])
        j = j + 1
    i = i + 1
inputs_x_array = np.array(inputs_x) 

# Instantiate a Gaussian Process model

## Matern, RBF, stationary_sigmoid
#kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale = [10., 100.], length_scale_bounds=[(1e-3, 1e3),(1e-4, 1e4)])


# manifold
#kernel = C(1.0, (0.01, 100)) \
#    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((1, 2),),
#                               transfer_fct="tanh", max_nn_weight=1)


# heteroscedastic    
prototypes = KMeans(n_clusters=8).fit(X).cluster_centers_
kernel = C(1.0, (1e-10, 1000)) * RBF(length_scale = [10., 100.], length_scale_bounds=[(1e-3, 1e3),(1e-4, 1e4)]) \
    + HeteroscedasticKernel.construct(prototypes, 1e-3, (1e-10, 50.0),
                                      gamma=1.0, gamma_bounds="fixed")
#gp.fit(X[:, np.newaxis], y)


gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y.reshape(-1,1)) #removing reshape results in a different error

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(inputs_x_array, return_std=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inputs_x_array[:,0],inputs_x_array[:,1],y_pred)
ax.scatter(X[:,0],X[:,1],y,color='orange')
ax.set_xlabel('X Label (radius)')
ax.set_ylabel('Y Label (time)')
ax.set_zlabel('Z Label (density)')
plt.show()

index_y1 = 3 #only valid until len_x2
print("Time is "+str(inputs_x_array[:,1][index_y1::len_x2][0])+"s")
plt.figure()
plt.scatter(inputs_x_array[:,0][index_y1::len_x2],y_pred[index_y1::len_x2]) #from x1_min to x1_max
plt.xlabel('X Label (radius)')
plt.ylabel('Z Label (density)') 
plt.show()

index_y2 = 7 #only valid until len_x1
print("Radius is "+str(inputs_x_array[:,0][index_y2*len_x2:index_y2*len_x2 + len_x2][0])+"m")
plt.figure()
plt.scatter(inputs_x_array[:,1][index_y2*len_x2:index_y2*len_x2 + len_x2],y_pred[index_y2*len_x2:index_y2*len_x2 + len_x2]) #from x2_min to x2_max
plt.xlabel('Y Label (time)')
plt.ylabel('Z Label (density)') 
plt.show()

print(gp.kernel_) #gives optimized hyperparameters 
print(gp.log_marginal_likelihood(gp.kernel_.theta)) #log-likelihood


alpha=1e-10
input_prediction = gp.predict(X,return_std=True)
K, K_gradient = gp.kernel_(X, eval_gradient=True)

K[np.diag_indices_from(K)] += alpha
L = cholesky(K, lower=True)  # Line 2
# Support multi-dimensional output of self.y_train_
if y.ndim == 1:
    y = y[:, np.newaxis]

alpha = cho_solve((L, True), y)
log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y, alpha)
log_likelihood_dims -= np.log(np.diag(L)).sum()
log_likelihood_dims -= (K.shape[0] / 2.) * np.log(2 * np.pi)
log_likelihood = log_likelihood_dims.sum(-1)
print(log_likelihood)

mean_sq_rel_err = ((input_prediction[0][:,0] - y[:,0])**2./y[:,0]**2.) #mean square relative error
print("Relative error on training data: "+str(np.mean(mean_sq_rel_err))+" +/- "+str(np.std(mean_sq_rel_err)))