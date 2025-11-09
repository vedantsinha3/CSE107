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


def mybilinear(y0: int, x0: int, y1: int, x1: int,
               f00: float, f01: float, f10: float, f11: float,
               y: float, x: float) -> float:
    """
    Bilinear interpolation given four integer pixel locations and their values
    around (y, x).

    Corners are:
      (y0, x0) -> f00, (y0, x1) -> f01,
      (y1, x0) -> f10, (y1, x1) -> f11

    (y, x) is the fractional location to interpolate.
    """
    # Guard against degenerate intervals
    if x1 == x0:
        tx = 0.0
    else:
        tx = (x - x0) / float(x1 - x0)
        if tx < 0.0:
            tx = 0.0
        elif tx > 1.0:
            tx = 1.0

    if y1 == y0:
        ty = 0.0
    else:
        ty = (y - y0) / float(y1 - y0)
        if ty < 0.0:
            ty = 0.0
        elif ty > 1.0:
            ty = 1.0

    top = f00 * (1.0 - tx) + f01 * tx
    bottom = f10 * (1.0 - tx) + f11 * tx
    value = top * (1.0 - ty) + bottom * ty
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
                # Determine integer neighbors with clamping to bounds
                if src_y <= 0:
                    y0 = y1 = 0
                elif src_y >= in_rows - 1:
                    y0 = y1 = in_rows - 1
                else:
                    y0 = int(np.floor(src_y))
                    y1 = y0 + 1

                if src_x <= 0:
                    x0 = x1 = 0
                elif src_x >= in_cols - 1:
                    x0 = x1 = in_cols - 1
                else:
                    x0 = int(np.floor(src_x))
                    x1 = x0 + 1

                f00 = float(inImage[y0, x0])
                f01 = float(inImage[y0, x1])
                f10 = float(inImage[y1, x0])
                f11 = float(inImage[y1, x1])
                outImage[out_r, out_c] = mybilinear(y0, x0, y1, x1, f00, f01, f10, f11, src_y, src_x)

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

