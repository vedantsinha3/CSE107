# Import pillow
from PIL import Image, ImageOps
# Import numpy
import numpy as np
from numpy import asarray

# Read the image from file.
im = Image.open('Beginnings.jpg')
# Show the image.
im.show()
# Convert image to gray scale.
im_gray = ImageOps.grayscale(im)
# Show the grayscale image.
im_gray.show()
# Convert image to numpy array
im_array = asarray(im_gray)

# Create numpy matrix of pixel values (same as im_array)
pixel_matrix = im_array

# Compute maximum pixel value using nested loops (no built-ins for max)
rows, cols = pixel_matrix.shape
max_val = 0
for r in range(rows):
    for c in range(cols):
        val = int(pixel_matrix[r][c])
        if r == 0 and c == 0:
            max_val = val
        elif val > max_val:
            max_val = val

print("Max of grayscale Beginnings image:", max_val)

# Rotate 90 degrees counterclockwise using nested loops
rot_ccw = np.zeros((cols, rows), dtype=pixel_matrix.dtype)
for r in range(rows):
    for c in range(cols):
        # Mapping for 90 CCW: (r, c) -> (cols - 1 - c, r)
        rot_ccw[cols - 1 - c][r] = pixel_matrix[r][c]

# Create image from CCW rotated matrix and display+save
im_ccw = Image.fromarray(rot_ccw)
im_ccw.show()
im_ccw.save('Beginnings_gray_rotated_ccw.jpg')

# Rotate 90 degrees clockwise using nested loops
rot_cw = np.zeros((cols, rows), dtype=pixel_matrix.dtype)
for r in range(rows):
    for c in range(cols):
        # Mapping for 90 CW: (r, c) -> (c, rows - 1 - r)
        rot_cw[c][rows - 1 - r] = pixel_matrix[r][c]

# Create image from CW rotated matrix and display+save
im_cw = Image.fromarray(rot_cw)
im_cw.show()
im_cw.save('Beginnings_gray_rotated_cw.jpg')

# Compute maximum pixel value of the clockwise rotated image using nested loops
rows_cw, cols_cw = rot_cw.shape
max_val_cw = 0
for r in range(rows_cw):
    for c in range(cols_cw):
        val = int(rot_cw[r][c])
        if r == 0 and c == 0:
            max_val_cw = val
        elif val > max_val_cw:
            max_val_cw = val

print("Max of clockwise rotated grayscale image:", max_val_cw)
