import matplotlib.pyplot as plt
import numpy as np

# Example data for the 3x3 tables (8 tables in total)
table_1 = np.array([
    ['', '0.5', ''],
    ['O.1', '1.0', '0.9'],
    ['', 'A', '']
])

table_2 = np.array([
    ['', '0.5', ''],
    ['1.1', 'A', '0.9'],
    ['', '', '']
])

table_3 = np.array([
    ['', '0.5', ''],
    ['0.1', '1.0', '0.9'],
    ['', 'A', '']
])

table_4 = np.array([
    ['', '0.5', ''],
    ['0.1', 'A', '1.9'],
    ['', '', '']
])

table_5 = np.array([
    ['', '0.5', ''],
    ['0.1', '1.0', '0.9'],
    ['', 'A', '']
])

table_6 = np.array([
    ['', '0.5', ''],
    ['0.6', 'A', '1.4'],
    ['', '', '']
])

table_7 = np.array([
    ['', '0.5', ''],
    ['0.1', '1.0', '0.9'],
    ['', 'A', '']
])

table_8 = np.array([
    ['', '0.8', ''],
    ['0.8', 'A', '0.9'],
    ['', '', '']
])

# Function to create a 3x3 table for each subplot
def create_table(ax, data):
    ax.axis('off')  # Hide axes
    table = ax.table(cellText=data, cellLoc='center', loc='center', colWidths=[0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    return table

# Create a figure with 8 subplots (2 columns, 4 rows)
fig, axs = plt.subplots(4, 2, figsize=(6, 8))  # Adjust the figsize to make the figure more compact

# List of all tables
tables = [table_1, table_2, table_3, table_4, table_5, table_6, table_7, table_8]

# Loop to create 8 tables in the subplots
for i in range(4):
    for j in range(2):
        # Choose the table from the list
        table_data = tables[i * 2 + j]
        # Create a table in the current subplot
        create_table(axs[i, j], table_data)

# Adjust layout for better spacing
plt.subplots_adjust(hspace=-4, wspace=0)  # Reduced vertical and horizontal spacing
plt.tight_layout(pad=-4)  # Adjust tight_layout to add padding
plt.show()
