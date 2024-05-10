import torch
import time

# Define a more complex function
def complex_function(x, y, z):
    a = torch.mm(x, y)
    b = torch.mm(a, z)
    c = torch.nn.functional.relu(b)
    d = c.mean(dim=1)
    return d

# Create some large matrices
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# Time the original function
start = time.time()
# Loop 1000 times
for _ in range(1000):
    complex_function(x, y, z)
end = time.time()
print(f"Original function took {end - start} seconds")

# Now let's JIT compile the function
complex_function_scripted = torch.jit.script(complex_function)

# Time the JIT-compiled function
start = time.time()
# Loop 1000 times
for _ in range(1000):
    complex_function_scripted(x, y, z)
end = time.time()
print(f"JIT-compiled function took {end - start} seconds")