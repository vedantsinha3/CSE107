# Import pillow
from PIL import Image, ImageOps
# Import numpy
import numpy as np
from numpy import asarray
# The size of the gradient image.
rows = 100
cols = 256
# Create a numpy matrix of this size.
im_pixels = np.zeros(shape=(rows, cols))

# Fill the matrix with a horizontal gradient 0..255 across columns for each row
for r in range(rows):
    for c in range(cols):
        im_pixels[r][c] = c

# Create image from numpy matrix (ensure uint8 type)
im_grad = Image.fromarray(np.uint8(im_pixels))

# Display the image
im_grad.show()

# Save the image as .tif
im_grad.save('Gradient.tif')

# Compute the average pixel value using nested loops (no built-ins)
total = 0
count = 0
for r in range(rows):
    for c in range(cols):
        total += int(im_pixels[r][c])
        count += 1

avg = total / count if count > 0 else 0
print("Average pixel value of gradient image:", avg)
