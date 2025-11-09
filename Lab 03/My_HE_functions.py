# MyHEFunctions.py

# Import numpy
import numpy as np

def compute_histogram( image_pixels ):

    # compute_histogram  Computes the normalized histogram (PMF) of a grayscale image.
    #
    # Syntax:
    #   hist = compute_histogram( image_pixels )
    #
    # Input:
    #   image_pixels = A 2D numpy array (float or int) with values in [0, 255].
    #
    # Output:
    #   hist = A length-256 numpy vector where hist[i] is the probability of intensity i.
    #
    # History:
    #   Implemented for Lab 03

    if image_pixels is None:
        raise ValueError("image_pixels must not be None")

    if not isinstance(image_pixels, np.ndarray):
        raise TypeError("image_pixels must be a numpy ndarray")

    if image_pixels.ndim != 2:
        raise ValueError("image_pixels must be a 2D grayscale image")

    # Ensure values are within [0, 255] and convert to integer indices
    clipped_pixels = np.clip(image_pixels, 0, 255)
    intensity_indices = clipped_pixels.astype(np.int32).ravel()

    # Count occurrences for each intensity using bincount for efficiency
    counts = np.bincount(intensity_indices, minlength=256).astype(np.float64)

    num_pixels = image_pixels.size
    if num_pixels == 0:
        # Avoid division by zero; return uniform zeros histogram
        return np.zeros(shape=(256,), dtype=np.float64)

    hist = counts / float(num_pixels)
    return hist

def equalize( in_image_pixels ):

    # equalize  Applies histogram equalization to a grayscale image and plots the
    #           intensity transformation function.
    #
    # Syntax:
    #   out_image_pixels = equalize( in_image_pixels )
    #
    # Input:
    #   in_image_pixels = A 2D numpy array (float or int) with values in [0, 255].
    #
    # Output:
    #   out_image_pixels = A 2D numpy array (float32) of the equalized image
    #                      with values in [0, 255].
    #
    # History:
    #   Implemented for Lab 03

    if in_image_pixels is None:
        raise ValueError("in_image_pixels must not be None")

    if not isinstance(in_image_pixels, np.ndarray):
        raise TypeError("in_image_pixels must be a numpy ndarray")

    if in_image_pixels.ndim != 2:
        raise ValueError("in_image_pixels must be a 2D grayscale image")

    # Compute the normalized histogram and its cumulative distribution function (CDF)
    hist = compute_histogram(in_image_pixels)
    cdf = np.cumsum(hist)

    # Create the intensity mapping: T(r) = round(255 * CDF(r))
    mapping = np.round(255.0 * cdf).astype(np.uint8)

    # Apply mapping to the input image
    clipped_pixels = np.clip(in_image_pixels, 0, 255).astype(np.int32)
    equalized_pixels_uint8 = mapping[clipped_pixels]

    # Convert to float32 to match lab expectations of float operations
    out_image_pixels = equalized_pixels_uint8.astype(np.float32)

    # Plot the transformation function T(r) vs r
    import matplotlib.pyplot as plt
    r_values = np.arange(256)
    plt.plot(r_values, mapping, linewidth=2)
    plt.title('Histogram Equalization Transformation')
    plt.xlabel('Input intensity r')
    plt.ylabel('Output intensity T(r)')
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

    return out_image_pixels

def plot_histogram( hist ):
    # plot_histgram  Plots the length 256 numpy vector representing the normalized
    # histogram of a grayscale image.
    #
    # Syntax:
    #   plot_histogram( hist )
    #
    # Input:
    #   hist = The length 256 histogram vector..
    #
    # Output:
    #   none
    #
    # History:
    #   S. Newsam     10/25/2025   created

    # Import plotting functions from matplotlib.
    import matplotlib.pyplot as plt

    plt.bar( range(256), hist )

    plt.xlabel('intensity value');

    plt.ylabel('PMF'); 

    plt.show()
