import jax.numpy as jnp
from jax import jit
import numpy as np
import time

# Create two large arrays
size = 10**2
a = np.arange(size)
b = np.arange(size)

# Define a manually batched function
def manually_batched_add1(a, b):
    output = np.empty_like(a)
    for i in range(len(a)):
        output[i] = a[i] + b[i]
    return output

# Define a vectorized function
def vectorized_add1(a, b):
    return a + b

# Time the manually batched function
start1 = time.time()
c = manually_batched_add1(a, b)
end1 = time.time()
print(f"Manually batched time: {end1 - start1}")

# Time the vectorized function
start2 = time.time()
c = vectorized_add1(a, b)
end2 = time.time()
print(f"Vectorized time: {end2 - start2}")


# JIT compile the functions
manually_batched_add2 = jit(manually_batched_add1)
vectorized_add2 = jit(vectorized_add1)

# Time the manually batched function
start = time.time()
c = manually_batched_add2(a, b).block_until_ready()  # block_until_ready is needed because JAX uses asynchronous execution by default
end = time.time()
print(f"Manually batched time: {end - start}")

# Time the vectorized function
start = time.time()
c = vectorized_add2(a, b).block_until_ready()
end = time.time()
print(f"Vectorized time: {end - start}")

