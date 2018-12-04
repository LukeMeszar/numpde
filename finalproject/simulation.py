from fenics import *
import numpy as np
from mshr import *
import matplotlib.pyplot as plt

mesh = Mesh("cad_files/files/Ferrari_SF71H_V2.xml")
coords = mesh.coordinates()
x = [a[0] for a in coords]
y = [a[1] for a in coords]
min_x = min(x)
max_x = max(x)
min_y = min(y)
max_y = max(y)
D = mesh.topology().dim()
print(D)
mf = MeshFunction("size_t", mesh, D - 1, mesh.domains())
plot(mesh)
plt.show()
