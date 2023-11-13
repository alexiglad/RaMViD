
import numpy as np
import matplotlib.pyplot as plt

# Load the npz file
data = np.load('logs/pred30x19x224x224x3.npz')

# Access the array 'arr_0'
images = data['arr_0']

# Assuming the images are in a format that Matplotlib can handle,
# you can display the first image as an example:
num_images = images[0].shape[0]

# Set up the subplots
rows = int(np.ceil(num_images / 4))
fig, axes = plt.subplots(rows, 4, figsize=(12, rows * 3))

# Flatten axes array for easy iteration
axes = axes.flatten()

# Display each image
for i in range(num_images):
    axes[i].imshow(images[6][i])
    axes[i].axis('off')  # Hide axis

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('logs/temp.png')
