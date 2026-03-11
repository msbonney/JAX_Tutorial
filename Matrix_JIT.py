"""
This is a demonstration of a simple functional relationship and how to use JAX with analytical functions.
- Created by Dr. Matthew Bonney
Swansea University

Tested on Python 3.11.9
To install packages:
pip install numpy jax matplotlib
"""
import numpy as np
import scipy.linalg as la
import jax
import jax.numpy as jnp
import time
#%% Create matrix and perform simple actions
# Let's use a simple 2DOF spring-mass system
# Numpy version
K_num = np.matrix([[100,-5],[-5,90]])
M_num = np.matrix([[5,0],[0,3]])
F_num_in = np.matrix([[5],[0]])
# Static Deformation
X_num = np.linalg.inv(K_num)*F_num_in
print('Displacement = ',str(X_num.flatten()))
# Eigen Analysis
Fn_num = np.sqrt(la.eigvals(K_num,M_num))
print('Using SciPy generalised Eigen : ',str(Fn_num))
# or using numpy eigvals
Fn_num = np.sqrt(np.linalg.eigvals(np.linalg.inv(M_num)*K_num))
print('Using Numpy Eigen : ',str(Fn_num))
#%% Now let's try JAX
K_jax = jnp.array([[100,-5],[-5,90]])
M_jax = jnp.array([[5,0],[0,3]])
F_jax = jnp.array([[5],[0]])
# Static Deformation
X_jax = jnp.linalg.inv(K_jax)*F_jax # Incorrect
print('Using the * operator : ',str(X_jax))
X_jax = jnp.matmul(jnp.linalg.inv(K_jax),F_jax) # Correct
print('Using the correct operator : ',str(X_jax))
# Let's check the output by creating a function
def staticDef(F_jax):
    K_jax = jnp.array([[100,-5],[-5,90]])
    return jnp.matmul(jnp.linalg.inv(K_jax),F_jax)
print(jax.make_jaxpr(staticDef)(F_jax))
#%% Now for dynamic JAX
# There is no generalised Eigenanalysis in jax currently
# There is some online code, but we can use theory instead
Fn_jax = jnp.sqrt(jnp.linalg.eigvals(jnp.linalg.inv(M_jax)*K_jax))
print('JAX produces frequencies of : ',str(Fn_jax),'\nNumpy produces frequencies of : ',str(Fn_num))
#%% All in One
# We can also create a function that takes in a material property to produce the natural frequencies that we can take derivatives of
def CompNat(A,E):
    # Use a 2 node compression bar for example to find the first Freqency
    # K = AE/L [[1,-1],[-1,1]]
    L = 0.12
    K = A*E/L*jnp.array([[1,-1],[-1,1]])
    # M = rho*A*L/2 [[1,0],[0,1]]
    rho = 2.7*10**-6
    M = rho*A*L/2*jnp.array([[1,0],[0,1]])
    Fn_jax = jnp.sqrt(jnp.linalg.eigvals(jnp.linalg.inv(M)*K))
    return jnp.real(jnp.min(Fn_jax)) # Needs to be real scalar output
print('Output Value of : ',str(CompNat(1,1000)))
# Now let's look at the gradients
# Inputs for grad need to be floats, not ints
print('Sensitivity to Area : ',str(jax.grad(CompNat,0)(1.0,1000.0))) 
print('Sensitivity to Youngs Modulus : ',str(jax.grad(CompNat,1)(1.0,1000.0))) 
#%% JIT
# Now that we can do it, let's look at scalability
tic = time.time()
Es = jnp.linspace(9e3,11e3,10000)
for E in Es:
    a = CompNat(1,E)
print('Time to complete Natively {}'.format(time.time()-tic))
# This compiles the function each time it is run
# We can precompile the function easily to reduce time
CompJit = jax.jit(CompNat)
tic = time.time()
for E in Es:
    a = CompJit(1,E)
print('Time to complete with JIT {}'.format(time.time()-tic))
# As a 1 to 1 comparison
tic = time.time()
CompJit = jax.jit(CompNat)
for E in Es:
    a = CompJit(1,E)
print('Time to complete with JIT + Compile {}'.format(time.time()-tic))