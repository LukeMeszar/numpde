from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

T = 1.0          # final time
num_steps = 250   # number of time steps
dt = T / num_steps # time step size
mu = 0.001         # dynamic viscosity
rho = 1            # density

PROGRESS = 16

# mesh = Mesh("airfoil_data/naca5012.xml")
# sub_domains = MeshFunction("size_t", mesh, "airfoil_data/naca5012_subdomains.xml")

mesh = Mesh("airfoil_data/naca5012_finer.xml")
sub_domains = MeshFunction("size_t", mesh, "airfoil_data/naca5012_finer_subdomains.xml")

V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)

# P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
# P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# TH = P2 * P1
# W = FunctionSpace(mesh, TH)

# inflow   = 'near(x[0], 0)'
# outflow  = 'near(x[0], 2.2)'
# walls    = 'near(x[1], 0) || near(x[1], 0.41)'
# cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

#inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
inflow_profile = ('0*x[1] + 1', '0')
noslip = Constant((0, 0))
# bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
# bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
# bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
# bcp_outflow = DirichletBC(Q, Constant(0), outflow)
# bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
# bcp = [bcp_outflow]


bcu_noslip = DirichletBC(V, noslip, sub_domains, 0) #top and bottom walls and airfoil
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), sub_domains, 1) #left walls
bcp_outflow = DirichletBC(Q, Constant(0), sub_domains, 2)
bcu = [bcu_noslip, bcu_inflow]
bcp = [bcp_outflow]



u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)



U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))


# dx = Measure("dx", domain=mesh, subdomain_data=sub_domains) # Volume integration
# ds = Measure("ds", domain=mesh, subdomain_data=sub_domains)


F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)


[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

xdmffile_u = XDMFFile('navier_stokes_airfoil/velocity_naca5012.xdmf')
xdmffile_p = XDMFFile('navier_stokes_airfoil/pressure_naca5012.xdmf')

timeseries_u = TimeSeries('navier_stokes_airfoil/velocity_series_naca5012')
timeseries_p = TimeSeries('navier_stokes_airfoil/pressure_series_naca5012')

File('navier_stokes_airfoil/naca5012.xml.gz') << mesh

progress = Progress('Time-stepping')
set_log_level(PROGRESS)

t = 0
counter = 0
for n in range(num_steps):
    counter += 1
    print("counter", counter)
    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'ilu')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'cg', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'bicgstab', 'ilu')

    # Plot solution
    plot(u_, title='Velocity')
    plot(p_, title='Pressure')

    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    # Update progress bar
    print('u max:', u_.vector().get_local().max())

plt.show()
