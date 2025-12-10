"""
Lab 04 Functions
Author: Vedant Sinha
Date: November 26, 2025

This module contains functions for linear spatial filtering and edge detection.
"""

import numpy as np


def spatial_filter(image, filter_mask):
    """
    Apply a spatial filter to a grayscale image using correlation.
    
    This function implements the correlation operation described in section 3.4.1
    of the textbook (3rd edition) or section 3.4 (4th edition). It uses zero 
    padding to handle boundaries.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image as a 2D numpy array
    filter_mask : numpy.ndarray
        Filter kernel as a 2D numpy array (assumed to be odd-sized)
    
    Returns:
    --------
    numpy.ndarray
        Filtered image with the same dimensions as the input image
    """
    # Get dimensions of the image and filter
    image_height, image_width = image.shape
    filter_height, filter_width = filter_mask.shape
    
    # Calculate padding needed (assuming odd-sized filter)
    pad_height = filter_height // 2
    pad_width = filter_width // 2
    
    # Create a zero-padded version of the image
    padded_image = np.pad(image, 
                          ((pad_height, pad_height), (pad_width, pad_width)), 
                          mode='constant', 
                          constant_values=0)
    
    # Initialize the output image
    filtered_image = np.zeros_like(image, dtype=np.float32)
    
    # Apply the filter using correlation
    # For each pixel in the output image
    for i in range(image_height):
        for j in range(image_width):
            # Extract the neighborhood from the padded image
            # The neighborhood is centered at position (i, j) in the original image
            neighborhood = padded_image[i:i+filter_height, j:j+filter_width]
            
            # Compute the correlation: element-wise multiplication and sum
            # This implements equation 3.43 (4th ed) or bottom of page 145 (3rd ed)
            filtered_image[i, j] = np.sum(neighborhood * filter_mask)
    
    return filtered_image


def gradient_magnitude(image):
    """
    Compute the gradient magnitude of an image using Sobel filters.
    
    This function computes the gradient magnitude by applying Sobel masks
    to detect horizontal and vertical edges, then computing the magnitude
    of the gradient vector.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image as a 2D numpy array
    
    Returns:
    --------
    numpy.ndarray
        Gradient magnitude image with the same dimensions as the input
    """
    # Define Sobel filters for computing gradients
    # Sobel mask for detecting vertical edges (gradient in x-direction)
    # From Figure 10.14 in the textbook
    sobel_x = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # Sobel mask for detecting horizontal edges (gradient in y-direction)
    # From Figure 10.14 in the textbook
    sobel_y = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    # Compute the gradient components using spatial filtering
    gradient_x = spatial_filter(image, sobel_x)
    gradient_y = spatial_filter(image, sobel_y)
    
    # Compute the magnitude of the gradient vector
    # Magnitude = sqrt(gx^2 + gy^2)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    return magnitude


def find_edges(image, threshold):
    """
    Detect edges in an image by thresholding the gradient magnitude.
    
    This function detects edges by computing the gradient magnitude using
    Sobel filters and then applying a threshold. Pixels with gradient 
    magnitude exceeding the threshold are marked as edge pixels (255),
    while others are marked as non-edge pixels (0).
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image as a 2D numpy array
    threshold : float
        Threshold value for edge detection
    
    Returns:
    --------
    numpy.ndarray
        Binary edge image with 255 at edge pixels and 0 elsewhere
    """
    # Compute the gradient magnitude
    grad_mag = gradient_magnitude(image)
    
    # Create a binary edge image by thresholding
    # Pixels with gradient magnitude > threshold are edges (255)
    # Pixels with gradient magnitude <= threshold are non-edges (0)
    edge_image = np.zeros_like(image, dtype=np.float32)
    edge_image[grad_mag > threshold] = 255
    
    return edge_image

