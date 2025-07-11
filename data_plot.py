# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
 
 
file_path = './user02_lab.csv'
data_df = pd.read_csv(file_path)
data_df = data_df[['x', 'y']]

data_df.head(), type(data_df), data_df['x'][0] 
 
fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# defining all 3 axis

_from =  0 
_to  = 1000

z =  list(range(len(data_df)))[_from: _to]   # np.linspace(0, 1, 100)
x = data_df['x'][_from: _to] #.tolist() # z * np.sin(25 * z)
y = data_df['y'][_from: _to] #.tolist()  # z * np.cos(25 * z)
c = z
# plotting
# ax.plot3D(z, x, y, 'green')
# ax.scatter(x, y, z, c = c, s = 1)
ax.set_xlabel('Z Label')
ax.set_ylabel('X Label')
ax.set_zlabel('Y Label')

ax.scatter(z, x, y, c = c, s = 1)
ax.set_title('3D scatter plot events')
plt.show()