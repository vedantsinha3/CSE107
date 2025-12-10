# my_hough_transform.py
#
# Implementation of the Hough transform for line detection.
# Created for CSE 107 Lab 5

# Import numpy
import numpy as np

# Import math library
import math


def my_hough_transform(i_edge):
    """
    my_hough_transform - Detect the most prominent line in an edge image using
    the Hough transform.
    
    The function implements the Hough transform by creating an accumulator
    matrix and voting for each edge point's possible line parameters.
    
    Syntax:
        theta_out, rho_out, accumulator = my_hough_transform(i_edge)
    
    Input:
        i_edge - An edge image (2D numpy array) with value 0 for non-edge points
                 and 255 for edge points.
    
    Output:
        theta_out - The angle (in degrees) the detected line makes with the 
                    vertical axis. Ranges from -89 to 90.
        rho_out - The length of the perpendicular bisector of the detected line.
        accumulator - The 2D accumulator matrix used in the Hough transform.
                      Rows correspond to rho values, columns to theta values.
    
    History:
        Created for CSE 107 Lab 5 - Hough Transform
    """
    
    # a) Determine the size of i_edge
    size_x, size_y = i_edge.shape
    
    # b) Create an empty accumulator matrix
    # Calculate the diagonal size of the image.
    # Rho can range from -D to D where D is the diagonal.
    D = math.floor(math.sqrt(size_x * size_x + size_y * size_y))
    
    # The accumulator matrix has:
    # - 2*D+1 rows: one for each possible rho value from -D to D
    # - 180 columns: one for each theta value from -89 to 90 degrees
    accumulator = np.zeros((2 * D + 1, 180), dtype=np.int32)
    
    # c) For every edge point in i_edge, plot the corresponding sinusoid
    # in the accumulator matrix.
    
    # Create array of theta values from -89 to 90 degrees (inclusive)
    theta_values = np.arange(-89, 91)
    
    # Pre-compute cosine and sine values for all theta (convert to radians first)
    theta_radians = np.radians(theta_values)
    cos_theta = np.cos(theta_radians)
    sin_theta = np.sin(theta_radians)
    
    # Iterate through all pixels in the edge image
    for x in range(size_x):
        for y in range(size_y):
            # Check if this is an edge point (value == 255)
            if i_edge[x, y] == 255:
                # For this edge point, compute rho for all possible theta values
                # using the normal form: x * cos(theta) + y * sin(theta) = rho
                rho_values = x * cos_theta + y * sin_theta
                
                # Round rho to nearest integer
                rho_values = np.round(rho_values).astype(int)
                
                # Vote in the accumulator for each (theta, rho) pair
                for i in range(180):
                    rho = rho_values[i]
                    
                    # Map rho to row index:
                    # rho ranges from -D to D
                    # row index = rho + D (maps -D to 0, D to 2*D)
                    row_idx = rho + D
                    
                    # Map theta to column index:
                    # theta ranges from -89 to 90
                    # column index = theta + 89 (maps -89 to 0, 90 to 179)
                    col_idx = i  # Since theta_values[i] = i - 89, col = (i-89)+89 = i
                    
                    # Ensure indices are within bounds and add a vote
                    if 0 <= row_idx < 2 * D + 1:
                        accumulator[row_idx, col_idx] += 1
    
    # d) Determine the cell with the most votes in the accumulator matrix
    # This gives us the theta and rho of the most prominent line.
    max_idx = np.argmax(accumulator)
    row_max, col_max = np.unravel_index(max_idx, accumulator.shape)
    
    # Convert accumulator indices back to theta and rho values
    # theta = column_index - 89 (reverses the mapping col = theta + 89)
    theta_out = col_max - 89
    
    # rho = row_index - D (reverses the mapping row = rho + D)
    rho_out = row_max - D
    
    # e) Return the computed values
    return theta_out, rho_out, accumulator

