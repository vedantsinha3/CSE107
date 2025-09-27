# Import pillow
from PIL import Image, ImageOps
# Import numpy
import numpy as np
from numpy import asarray
# Read the image from file.
im = Image.open('Tree.tif')
# Show the image.
im.show()
# Print the image mode.
print("image mode is:", im.mode)
# Create numpy matrix to access the pixel values.
im_pixels = asarray(im)
# Import myImageInverse from myImageInverse
from MyImageFunctions import myImageInverse
im_inv_pixels = myImageInverse(im_pixels)
# Create an image from im_inv_pixels.
im_inv = Image.fromarray(np.uint8(im_inv_pixels))
# Show the inverse image.
im_inv.show()
# Save the inverse image to a file.
im_inv.save("Tree_inv.tif")

# Compute maximum pixel value of the inverse image using nested loops (no built-ins)
rows_i, cols_i = im_inv_pixels.shape
max_inv = 0
for r in range(rows_i):
    for c in range(cols_i):
        val = int(im_inv_pixels[r][c])
        if r == 0 and c == 0:
            max_inv = val
        elif val > max_inv:
            max_inv = val
print("Max of inverse Tree image:", max_inv)

# Also compute the minimum of the original image using nested loops (for relationship)
rows_o, cols_o = im_pixels.shape
min_orig = 0
for r in range(rows_o):
    for c in range(cols_o):
        val = int(im_pixels[r][c])
        if r == 0 and c == 0:
            min_orig = val
        elif val < min_orig:
            min_orig = val
print("Min of original Tree image:", min_orig)
print("255 - min(original) =", 255 - min_orig)