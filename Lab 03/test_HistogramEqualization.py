# Import pillow
from PIL import Image, ImageOps

# Import numpy
import numpy as np
from numpy import asarray

###############################################################################
# Perform histogram equalization on the dark image.
###############################################################################

# Read the dark image from file.
dark_im = Image.open('Lab_03_image1_dark.tif')

# Show the image.
dark_im.show()

# Create numpy matrix to access the pixel values.
# NOTE THAT WE WE ARE CREATING A FLOAT32 ARRAY SINCE WE WILL BE DOING
# FLOATING POINT OPERATIONS IN THIS LAB.
dark_im_pixels = asarray(dark_im, dtype=np.float32)

# Import compute_histogram from My_HE_functions.
from My_HE_functions import compute_histogram

# Compute the histogram of the dark image.
dark_hist = compute_histogram( dark_im_pixels )

# Import plot_histogram from My_HE_functions.
from My_HE_functions import plot_histogram

# Plot the histogram for the dark image.
plot_histogram( dark_hist )

print('Dark image has mean = %f and standard deviation = %f' % \
    (np.mean(dark_im_pixels), np.std(dark_im_pixels)))

# Import equalize from My_HE_functions.
from My_HE_functions import equalize

# Apply histogram equalization to the dark image.
equalized_dark_im_pixels = equalize( dark_im_pixels );

# Create an image from numpy matrix equalized_dark_image_pixels.
equalized_dark_image = Image.fromarray(np.uint8(equalized_dark_im_pixels.round()))

# Show the equalized image.
equalized_dark_image.show()

# Save the equalized image.
equalized_dark_image.save('equalized_dark_image.tif');

# Compute the histogram of the equalized dark image.
equalized_dark_hist = compute_histogram( equalized_dark_im_pixels )

# Plot the histogram for the equalized dark image.
plot_histogram( equalized_dark_hist )

print('Equalized dark image has mean = %f and standard deviation = %f' % \
    (np.mean(equalized_dark_im_pixels), np.std(equalized_dark_im_pixels)))

###############################################################################
# Perform histogram equalization on the light image.
###############################################################################

# Read the light image from file.
light_im = Image.open('Lab_03_image2_light.tif')

# Show the image.
light_im.show()

# Create numpy matrix to access the pixel values.
# NOTE THAT WE WE ARE CREATING A FLOAT32 ARRAY SINCE WE WILL BE DOING
# FLOATING POINT OPERATIONS IN THIS LAB.
light_im_pixels = asarray(light_im, dtype=np.float32)

# Compute the histogram of the light image.
light_hist = compute_histogram( light_im_pixels )

# Plot the histogram for the light image.
plot_histogram( light_hist )

print('\nLight image has mean = %f and standard deviation = %f' % \
    (np.mean(light_im_pixels), np.std(light_im_pixels)))

# Apply histogram equalization to the light image.
equalized_light_im_pixels = equalize( light_im_pixels );

# Create an image from numpy matrix equalized_light_image_pixels.
equalized_light_image = Image.fromarray(np.uint8(equalized_light_im_pixels.round()))

# Show the equalized image.
equalized_light_image.show()

# Save the equalized image.
equalized_light_image.save('equalized_light_image.tif');

# Compute the histogram of the equalized light image.
equalized_light_hist = compute_histogram( equalized_light_im_pixels )

# Plot the histogram for the equalized light image.
plot_histogram( equalized_light_hist )

print('Equalized light image has mean = %f and standard deviation = %f' % \
    (np.mean(equalized_light_im_pixels), np.std(equalized_light_im_pixels)))

###############################################################################
# Perform histogram equalization on the low contrast image.
###############################################################################

# Read the low contrast image from file.
low_contrast_im = Image.open('Lab_03_image3_low_contrast.tif')

# Show the image.
low_contrast_im.show()

# Create numpy matrix to access the pixel values.
# NOTE THAT WE WE ARE CREATING A FLOAT32 ARRAY SINCE WE WILL BE DOING
# FLOATING POINT OPERATIONS IN THIS LAB.
low_contrast_im_pixels = asarray(low_contrast_im, dtype=np.float32)

# Compute the histogram of the low contrast image.
low_contrast_hist = compute_histogram( low_contrast_im_pixels )

# Plot the histogram for the low contrast image.
plot_histogram( low_contrast_hist )

print('Low contrast image has mean = %f and standard deviation = %f' % \
    (np.mean(low_contrast_im_pixels), np.std(low_contrast_im_pixels)))

# Apply histogram equalization to the low contrast image.
equalized_low_contrast_im_pixels = equalize( low_contrast_im_pixels );

# Create an image from numpy matrix equalized_low_contrast_image_pixels.
equalized_low_contrast_image = Image.fromarray(np.uint8(equalized_low_contrast_im_pixels.round()))

# Show the equalized image.
equalized_low_contrast_image.show()

# Save the equalized image.
equalized_low_contrast_image.save('equalized_low_contrast_image.tif');

# Compute the histogram of the equalized low contrast image.
equalized_low_contrast_hist = compute_histogram( equalized_low_contrast_im_pixels )

# Plot the histogram for the equalized low contrast image.
plot_histogram( equalized_low_contrast_hist )

print('Equalized low contrast image has mean = %f and standard deviation = %f' % \
    (np.mean(equalized_low_contrast_im_pixels), np.std(equalized_low_contrast_im_pixels)))

###############################################################################
# Perform histogram equalization on the high contrast image.
###############################################################################

# Read the high contrast image from file.
high_contrast_im = Image.open('Lab_03_image4_high_contrast.tif')

# Show the image.
high_contrast_im.show()

# Create numpy matrix to access the pixel values.
# NOTE THAT WE WE ARE CREATING A FLOAT32 ARRAY SINCE WE WILL BE DOING
# FLOATING POINT OPERATIONS IN THIS LAB.
high_contrast_im_pixels = asarray(high_contrast_im, dtype=np.float32)

# Compute the histogram of the high contrast image.
high_contrast_hist = compute_histogram( high_contrast_im_pixels )

# Plot the histogram for the high contrast image.
plot_histogram( high_contrast_hist )

print('High contrast image has mean = %f and standard deviation = %f' % \
    (np.mean(high_contrast_im_pixels), np.std(high_contrast_im_pixels)))

# Apply histogram equalization to the high contrast image.
equalized_high_contrast_im_pixels = equalize( high_contrast_im_pixels );

# Create an image from numpy matrix equalized_high_contrast_image_pixels.
equalized_high_contrast_image = Image.fromarray(np.uint8(equalized_high_contrast_im_pixels.round()))

# Show the equalized image.
equalized_high_contrast_image.show()

# Save the equalized image.
equalized_high_contrast_image.save('equalized_high_contrast_image.tif');

# Compute the histogram of the equalized high contrast image.
equalized_high_contrast_hist = compute_histogram( equalized_high_contrast_im_pixels )

# Plot the histogram for the equalized high contrast image.
plot_histogram( equalized_high_contrast_hist )

print('Equalized high contrast image has mean = %f and standard deviation = %f' % \
    (np.mean(equalized_high_contrast_im_pixels), np.std(equalized_high_contrast_im_pixels)))
