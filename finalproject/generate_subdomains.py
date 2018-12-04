from dolfin import *
import sys

set_log_level(1)
# Read mesh
ifile = sys.argv[1]
ofile = sys.argv[2]
ofile2 = sys.argv[3]
mesh = Mesh(ifile)
coords = mesh.coordinates()
x = [a[0] for a in coords]
y = [a[1] for a in coords]
min_x = min(x)
max_x = max(x)
min_y = min(y)
max_y = max(y)

# Sub domain for no-slip (mark whole boundary, inflow and outflow will overwrite)
class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Sub domain for inflow (right)
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], min_x) and on_boundary

# Sub domain for outflow (left)
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], max_x) and on_boundary


# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
#sub_domains_bool = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
#sub_domains_double = MeshFunction("double", mesh, mesh.topology().dim() - 1)

# Mark all facets as sub domain 3
sub_domains.set_all(3)
#sub_domains_bool.set_all(False)
#sub_domains_double.set_all(0.3)

# Mark no-slip facets as sub domain 0, 0.0
noslip = Noslip()
noslip.mark(sub_domains, 0)
#noslip.mark(sub_domains_double, 0.0)

# Mark inflow as sub domain 1, 01
inflow = Inflow()
inflow.mark(sub_domains, 1)
#inflow.mark(sub_domains_double, 0.1)

# Mark outflow as sub domain 2, 0.2, True
outflow = Outflow()
outflow.mark(sub_domains, 2)
#outflow.mark(sub_domains_double, 0.2)
#outflow.mark(sub_domains_bool, True)

# Save sub domains to file
file = File(ofile)
file << sub_domains



# Save sub domains to VTK files
file = File(ofile2)
file << sub_domains
#
# file = File("subdomains_double.pvd")
# file << sub_domains_double
