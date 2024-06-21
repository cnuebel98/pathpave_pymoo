import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.ref_dirs import get_reference_directions

# Generate reference directions using Das-Dennis method
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=10)

# Print the generated reference directions
print("Reference Directions:")
print(ref_dirs)

# Plot the reference directions
plt.scatter(ref_dirs[:, 0], ref_dirs[:, 1], color='red')
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.title("Reference Directions using Das-Dennis Method")
plt.grid(True)
plt.show()