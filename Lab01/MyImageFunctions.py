import numpy as np

def myImageInverse(inImage):
    """
    Takes a numpy matrix representing a grayscale image (dtype uint8) and
    returns a numpy matrix of the same size that is the image inverse,
    computed as 255 - in_value using nested loops.
    """
    # Determine size
    rows, cols = inImage.shape
    # Create output matrix of same size and dtype
    outImage = np.zeros((rows, cols), dtype=inImage.dtype)
    # Compute inverse using nested loops
    for r in range(rows):
        for c in range(cols):
            in_val = int(inImage[r][c])
            out_val = 255 - in_val
            outImage[r][c] = out_val
    return outImage


