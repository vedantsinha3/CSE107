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


def mybilinear(inImage: np.ndarray, y: float, x: float) -> float:
    """
    Bilinear interpolation of a grayscale image at fractional row y and column x.
    Returns a float value. Coordinates are clamped to image bounds.
    """
    num_rows, num_cols = inImage.shape

    # Handle out-of-bounds by clamping to the border and using zero fractional part
    if y <= 0:
        y0 = y1 = 0
        dy = 0.0
    elif y >= num_rows - 1:
        y0 = y1 = num_rows - 1
        dy = 0.0
    else:
        y0 = int(np.floor(y))
        y1 = y0 + 1
        dy = y - y0

    if x <= 0:
        x0 = x1 = 0
        dx = 0.0
    elif x >= num_cols - 1:
        x0 = x1 = num_cols - 1
        dx = 0.0
    else:
        x0 = int(np.floor(x))
        x1 = x0 + 1
        dx = x - x0

    top_left = float(inImage[y0, x0])
    top_right = float(inImage[y0, x1])
    bottom_left = float(inImage[y1, x0])
    bottom_right = float(inImage[y1, x1])

    top = top_left * (1.0 - dx) + top_right * dx
    bottom = bottom_left * (1.0 - dx) + bottom_right * dx
    value = top * (1.0 - dy) + bottom * dy
    return float(value)


def myImageResize(inImage: np.ndarray, out_rows: int, out_cols: int, method: str) -> np.ndarray:
    """
    Resize a grayscale image (numpy ndarray) to (out_rows, out_cols) using
    either 'nearest' or 'bilinear' interpolation. Returns float32 ndarray.
    """
    if inImage.ndim != 2:
        raise ValueError("myImageResize expects a 2D grayscale image array")

    in_rows, in_cols = inImage.shape
    if out_rows <= 0 or out_cols <= 0:
        raise ValueError("Output dimensions must be positive")

    outImage = np.zeros((out_rows, out_cols), dtype=np.float32)

    # Scale from output pixel centers to input coordinates
    scale_y = in_rows / float(out_rows)
    scale_x = in_cols / float(out_cols)

    use_bilinear = method.lower() == 'bilinear'
    use_nearest = method.lower() == 'nearest'
    if not (use_bilinear or use_nearest):
        raise ValueError("method must be 'nearest' or 'bilinear'")

    for out_r in range(out_rows):
        src_y = (out_r + 0.5) * scale_y - 0.5
        for out_c in range(out_cols):
            src_x = (out_c + 0.5) * scale_x - 0.5

            if use_nearest:
                nn_y = int(round(src_y))
                nn_x = int(round(src_x))
                # Clamp to valid bounds
                if nn_y < 0:
                    nn_y = 0
                elif nn_y >= in_rows:
                    nn_y = in_rows - 1
                if nn_x < 0:
                    nn_x = 0
                elif nn_x >= in_cols:
                    nn_x = in_cols - 1
                outImage[out_r, out_c] = float(inImage[nn_y, nn_x])
            else:
                outImage[out_r, out_c] = mybilinear(inImage, src_y, src_x)

    return outImage


def myRMSE(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error between two equally sized images.
    Returns a Python float.
    """
    if imageA.shape != imageB.shape:
        raise ValueError("Images must have the same shape for RMSE computation")
    diff = imageA.astype(np.float64) - imageB.astype(np.float64)
    mse = np.mean(diff * diff)
    return float(np.sqrt(mse))


def myimresize(inImage: np.ndarray, out_rows: int, out_cols: int, method: str) -> np.ndarray:
    """
    Lowercase alias matching some specs; forwards to myImageResize.
    """
    return myImageResize(inImage, out_rows, out_cols, method)

