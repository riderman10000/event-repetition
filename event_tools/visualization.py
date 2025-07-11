import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties

import numpy as np
import pandas as pd 

# Disassemble each column in DF
def get_x_y_t(df):
    # The order of time stamp unfolds
    xs =  df['timestamp'] - df['timestamp'][0] # np.arange(df.shape[0]) #
    ys = df['y']
    zs = df['x']
    return xs,ys,zs

def set_label(ax):
    ax.set_xlabel('T',fontsize=20)
    ax.set_ylabel('Y',fontsize=20)
    ax.set_zlabel('X',fontsize=20,labelpad=0)
    
 # Draw 3D graphics
def plot_3D(file_name,class_num=None,delta=5,count=100,mean_count=100):
    fig = plt.figure() # figsize=(23,12),dpi=800)
    font = FontProperties(family='Times New Roman',size=20)
    ax = fig.add_subplot(121, projection='3d')
    # Original 3D picture
    df = pd.read_csv(f'./event_csv/split_data/class{class_num}/{file_name}')
    xs,ys,zs = get_x_y_t(df)
    colors = np.linspace(0, 1, len(xs))
    ax.scatter(xs, ys, zs, s=2, c=colors, cmap='viridis', label='Event')
    plt.title(f'Original event stream',size=20)
    set_label(ax)
    
    # ax = fig.add_subplot(122, projection='3d') 
    # # 3D diagram after compression
    # df = pd.read_csv(f'../../event_csv/compress_event_manhattan/class{class_num}/{file_name}')
    # xs,ys,zs = get_x_y_t(df)
    # ax.scatter(xs, ys, zs, s=2,c='red',label='Event')
    # plt.title(f'Event stream is processed by Δ={delta},count={count}',size=20)
    # set_label(ax)
    plt.show()
    
    
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


if __name__ == "__main__":
    plot_3D('user02_lab.csv',class_num=2)
    plt.show()