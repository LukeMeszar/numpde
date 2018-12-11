from fenics import *
import numpy as np
from mshr import *
import matplotlib.pyplot as plt


airfoil = Mesh("airfoil_data/ag04.xml")
domain = Rectangle(Point(-0.5, -0.5), Point(1.5, 0.5))

mesh = generate(domain,1)
#mf = MeshFunction("size_t", mesh, D - 1, mesh.domains())
plot(mesh)
plt.show()
