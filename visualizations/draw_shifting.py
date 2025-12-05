import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots()

# Customize tick labels and grid for a 3x3 grid
ax.set_xticks(np.arange(1, 3, 1), minor=True)
ax.set_yticks(np.arange(1, 3, 1), minor=True)
ax.grid(True, which='both', color='black', linestyle='-', linewidth=2)

# Hide major ticks and labels
ax.tick_params(which='major', size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

# Set plot limits to show only the outer border of the 3x3 grid
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)

plt.show()
