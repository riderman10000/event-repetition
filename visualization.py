import cv2 
import copy 
import numpy as np 

def plot_heatmap(dist: np.ndarray, log_scale: bool = False) -> np.ndarray:
    """Plot the temporal self-similarity matrix into an OpenCV image."""
    dist = copy.deepcopy(dist)
    np.fill_diagonal(dist, np.nan)
    if log_scale:
        dist = np.log(1 + dist)
    dist = -dist # Invert the distance
    zmin, zmax = np.nanmin(dist), np.nanmax(dist)
    heatmap = (dist - zmin) / (zmax - zmin) # Normalize into [0, 1]
    heatmap = np.nan_to_num(heatmap, nan=1)
    heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    return heatmap

def show(image, name='test', wait= 0 ):
    cv2.imshow(name, image) 
    k = cv2.waitKey(wait)
    cv2.destroyAllWindows()
    return k 