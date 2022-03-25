from matplotlib import pyplot as plt
import shapely.geometry
import numpy as np

p = shapely.geometry.Point(0,0,0)
plt.figure(figsize=(5,5))
x = p.buffer(1)
c = np.array(x.boundary.coords)
print(c)
plt.plot(c[:,0],c[:,1])
plt.show()