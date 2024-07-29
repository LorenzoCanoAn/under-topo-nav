import pyvista as pv
import numpy as np

n1 = (-32, -114, 0)
n2 = (27, 101, 0)
n3 = (-135, 151, 0)
n4 = (142, 494, 0)
n5 = (-51, 633, 0)
n6 = (287, 1005, 0)
n7 = (154, 1045, 0)
n8 = (405, 1418, 0)
n9 = (305, 1458, 0)
n10 = (500, 1745, 0)
n11 = (489, 1784, 0)
n12 = (524, 1819, 0)

e1 = (n1, n2)
e2 = (n2, n3)
e3 = (n2, n4)
e4 = (n4, n5)
e5 = (n4, n6)
e6 = (n6, n7)
e7 = (n6, n8)
e8 = (n8, n9)
e9 = (n8, n10)
e10 = (n10, n11)
e11 = (n10, n12)

nodes = (n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12)
edges = (e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11)

plotter = pv.Plotter()
for n in nodes:
    print(n)
    plotter.add_mesh(pv.Sphere(radius=15, center=n), color="r")
for edge in edges:
    n1, n2 = edge
    n1 = np.array(n1)
    n2 = np.array(n2)
    direction = n1 - n2
    height = np.linalg.norm(direction, 2)
    center = (n1 + n2) / 2
    plotter.add_mesh(pv.Cylinder(center=center, direction=direction, radius=7.5, height=height))
plotter.show()
