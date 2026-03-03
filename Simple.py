"""
This is a demonstration of a simple functional relationship and how to use JAX with analytical functions.
- Created by Dr. Matthew Bonney
Swansea University

Tested on Python 3.11.9
To install packages:
pip install numpy jax matplotlib
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
#%% Setup of function and check
# We will be using exp(-t)*sin(4t)
# First let's set up Numpy version
def num_fun(t):
    return np.exp(-t)*np.sin(4*t)
def num_jac(t):
    a = -1*np.exp(-t)*np.sin(4*t)
    b = 4* np.exp(-t)*np.cos(4*t)
    return a+b
# Now the JAX version
def jax_fun(t):
    a = jnp.exp(-t)
    b = jnp.sin(4*t)
    return a*b
def jax_short(t):
    return jnp.exp(-t)*jnp.sin(4*t)
#%% Let's investigate the steps
print(jax.make_jaxpr(jax_fun)(1.0))
#Compare to shorter version
print(jax.make_jaxpr(jax_short)(1.0))
#%% Let's test out for t=1
t=1.0
numpy_value = num_fun(t)
numpy_jac_value = num_jac(t)
jax_value = jax_fun(t)
jax_jac_value = jax.grad(jax_fun)(t)
print('Numpy results in value of {:f} and derivative of {:f}\n'.format(numpy_value,numpy_jac_value))
print('JAX results in value of {:f} and derivative of {:f}\n'.format(jax_value,jax_jac_value))
#%% Arrays and plotting
x = np.arange(0,4,0.01)
x_j = jnp.arange(0.,4.,0.01)
# Numpy values
y = num_fun(x)
y_d = num_jac(x)
# JAX values
y_j = jax_fun(x_j)
# jax.grad cannot take in array inputs
y_j_d_n = np.zeros(len(x_j))
y_j_d_j = jnp.zeros(len(x_j))
# jnp arrays are immutable - 2 ways to set this up
for i in range(len(x_j)):
    y_j_d_n[i] = jax.grad(jax_fun)(x_j[i]) # convert to numpy array
    y_j_d_j = y_j_d_j.at[i].set(jax.grad(jax_fun)(x_j[i])) # keep jnp array


# Plot nominal values
plt.figure()
plt.plot(x,y,'k-')
plt.plot(x_j,y_j,'r--')
plt.legend(['Numpy','JAX'])
plt.xlabel('Time [s]')
plt.ylabel('Response')

# Plot derivative values
plt.figure()
plt.plot(x,y_d,'k-')
plt.plot(x_j,y_j_d_n,'r--')
plt.plot(x_j,y_j_d_j,'g--')
plt.legend(['Numpy','JAX - Numpy','JAX'])
plt.xlabel('Time [s]')
plt.ylabel('Jacobian Response')
