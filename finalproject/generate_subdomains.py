from dolfin import *
import sys

set_log_level(1)
# Read mesh
ifile = sys.argv[1]
ofile = sys.argv[2]
if len(sys.argv) > 3:
    ofile2 = sys.argv[3]
else:
    ofile2 = None
mesh = Mesh(ifile)
coords = mesh.coordinates()
x = [a[0] for a in coords]
y = [a[1] for a in coords]
min_x = min(x)
max_x = max(x)
min_y = min(y)
max_y = max(y)
print(min_x,max_x)

# Sub domain for no-slip (mark whole boundary, 
#inflow and outflow will overwrite)
class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Sub domain for inflow (right)
class TopWall(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], max_y) and on_boundary

class BotWall(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], min_y) and on_boundary

class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], min_x) and on_boundary

# Sub domain for outflow (left)
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], max_x) and on_boundary

# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

# Mark all facets as sub domain 3
sub_domains.set_all(4)

# Mark no-slip facets as sub domain 0
noslip = Noslip()
noslip.mark(sub_domains, 0)

# Mark inflow as sub domain 1
inflow = Inflow()
inflow.mark(sub_domains, 1)

# Mark outflow as sub domain 2
outflow = Outflow()
outflow.mark(sub_domains, 2)

topwall = TopWall()
topwall.mark(sub_domains,3)

botwall = BotWall()
botwall.mark(sub_domains,3)

# Save sub domains to file
file = File(ofile)
file << sub_domains

# Save sub domains to VTK files
if ofile2 != None:
    file = File(ofile2)
    file << sub_domains
