# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, debug=True, cpu_max_num_threads=1, advanced_optimization=False,  kernel_profiler=True)

dim = 3
num_particles = 8000
h = 0.01


p_rho0 = 1000.0
particle_radius = h/3
p_vol =  (particle_radius)**2 
p_mass = p_vol * p_rho0
time_delta = 1e-4


positions = ti.Vector.field(dim, float, num_particles)
old_positions = ti.Vector.field(dim, float, num_particles)
velocities = ti.Vector.field(dim, float, num_particles)


paused = ti.field(dtype=ti.i32, shape=())


# ---------------------------------------------------------------------------- #
#                                Neighbor search                               #
# ---------------------------------------------------------------------------- #
# the helper function for neighbor search
@ti.func
def get_cell_ID(pos):
    return int(pos * cell_inv)

cell_size = 3 * h
cell_inv = 1.0/cell_size
num_cell_x = num_cell_y = num_cell_z = (int)(cell_inv)

max_num_particles_per_cell = 100
grid_num_particles = ti.field(ti.i32, shape=(num_cell_x,num_cell_y,num_cell_z))
grid2particles = ti.field(ti.i32, shape=(num_cell_x,num_cell_y,num_cell_z, max_num_particles_per_cell))

max_num_neighbors = 30
num_neighbor = ti.field(ti.i32, shape=num_particles)
particle_neighbors = ti.field(ti.i32, shape=(num_particles, max_num_neighbors))

@ti.func
def clear_lists():
    for cell in ti.grouped(grid_num_particles):
        grid_num_particles[cell] = 0

@ti.func
def build_parInCell_list():
    for p_i in range(num_particles):
        cell = get_cell_ID(positions[p_i])
        offs = grid_num_particles[cell]
        grid_num_particles[cell] += 1
        grid2particles[cell, offs] = p_i

pos_ij = ti.field(ti.f32, (num_particles, max_num_neighbors))
@ti.func     
def build_neighbor_list():
    neighbor_radius = 1.1 * h
    for p_i in positions:
        cell = get_cell_ID(positions[p_i])
        nb_i = 0 # the nb_i th neighbor of p_i
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1,2)))):
            for j in range(grid_num_particles[cell + offs]):
                p_j = grid2particles[cell+ offs, j] #particles in the neighbor cell
                dist=  (positions[p_i] - positions[p_j]).norm()
                if dist < neighbor_radius:
                    particle_neighbors[p_i, nb_i] = p_j
                    nb_i += 1
                    num_neighbor[p_i] +=1
                    pos_ij[p_i, nb_i] = dist


@ti.func
def neighbor_search():
    clear_lists()
    build_parInCell_list()
    build_neighbor_list()

# ---------------------------------------------------------------------------- #
#                              end neighbor search                             #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    proloug                                   #
# ---------------------------------------------------------------------------- #
@ti.kernel
def prologue():
    for i in positions:
        old_positions[i] = positions[i]
    extern_force()
    neighbor_search()
ti.print_kernel_profile_info('count')

@ti.func
def extern_force():
    for i in positions:
        g = ti.Vector([0., -0.1, 0.1])
        vel = velocities[i]
        vel += g * time_delta
        positions[i] += vel * time_delta

@ti.func
def boundary_handling(p_i):
    padding = 1e-1
    bmin = 0. + padding
    bmax = 1. - padding
    for dim_i in ti.static(range(dim)):
        if positions[p_i][dim_i] > bmax:
            positions[p_i][dim_i] = bmax - ti.random() * 1e-3
            velocities[p_i][dim_i] *= -1.0 
        if positions[p_i][dim_i] < bmin:
            positions[p_i][dim_i] = bmin + ti.random() * 1e-3
            velocities[p_i][dim_i] *= -1.0 

@ti.kernel
def epilogue():
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    for i in positions:
        boundary_handling(i)
# ---------------------------------------------------------------------------- #
#                                    kernels                                   #
# ---------------------------------------------------------------------------- #
import math
poly6_factor = 315.0 / 64.0 / math.pi
@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

spiky_grad_factor = 15.0 / math.pi
@ti.func
def spiky_gradient(r, h):
    result = 0.0 
    if 0 < r and r < h:
        x = (h - r) / (h * h)
        result = spiky_grad_factor * x * x * x
    return result


# ---------------------------------------------------------------------------- #
#                                compute lambdas                               #
# ---------------------------------------------------------------------------- #
# compute lambdas, see Eq(11)
lambdas = ti.field(ti.f32, num_particles)
lambda_epsilon = 100.0
@ti.func
def compute_lambda():
    for p_i in positions:
        grad_sqr_sum = compute_grad_constraint(p_i)
        density_constraint=(compute_density(p_i) / p_rho0) - 1.0
        lambdas[p_i] = (-density_constraint) / (grad_sqr_sum + lambda_epsilon)
    

@ti.func
def compute_grad_constraint(p_i):
    sum_gradient_sqr = 0.0
    grad_i = ti.Vector([0.0, 0.0, 0.0])
    grad_j = ti.Vector([0.0, 0.0, 0.0])
    for nb_i in range(num_neighbor[p_i]):
        p_j = particle_neighbors[p_i, nb_i]
        dir = (positions[p_i]- positions[p_j])/pos_ij[p_i, nb_i]
        grad_j = spiky_gradient(pos_ij[p_i,nb_i], h) * dir
        grad_i += grad_j
        sum_gradient_sqr += grad_j.dot(grad_j)
    sum_gradient_sqr += grad_i.dot(grad_i)
    return sum_gradient_sqr

@ti.func
def compute_density(p_i):
    density_i = 0.0
    for nb_i in range(num_neighbor[p_i]):
        density_i +=  poly6_value(pos_ij[p_i,nb_i], h) * p_mass
    return density_i

# ---------------------------------------------------------------------------- #
#                              end compute lambdas                             #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                            compute position deltas                           #
# ---------------------------------------------------------------------------- #
# compute position deltas, see Eq(14) 
position_deltas = ti.Vector.field(dim, ti.f32, num_particles)
@ti.func
def compute_position_delta():
    for p_i in positions:
        res = ti.Vector([0.0, 0.0, 0.0])
        for nb_i in range(num_neighbor[p_i]):
            p_j = particle_neighbors[p_i, nb_i]
            scorr_ij = compute_scorr(pos_ij[p_i,nb_i])
            res += (lambdas[p_i] + lambdas[p_j] + scorr_ij) * spiky_gradient(pos_ij[p_i,nb_i], h)
        position_deltas[p_i] = res / p_rho0

# s_corr, see Eq(13) 
corr_deltaQ = 0.3 * h
corr_k = 0.1
@ti.func
def compute_scorr(dist):
    x = poly6_value(dist, h) / poly6_value(corr_deltaQ, h)
    x = x * x * x * x
    return (-corr_k) * x
# ---------------------------------------------------------------------------- #
#                          end compute position deltas                         #
# ---------------------------------------------------------------------------- #

@ti.kernel
def substep():
    compute_lambda()
    compute_position_delta()
    # update positions
    for i in positions:
        positions[i] += position_deltas[i]


solverIterations = 4 # the paper suggests 2-4
def run_pbf():
    prologue()
    for _ in range(solverIterations):  
        substep()
    epilogue()

@ti.kernel
def init():
    init_pos = ti.Vector([0.2,0.3,0.2])
    cube_size = 0.4
    spacing = 0.02
    num_per_row = (int) (cube_size // spacing)
    num_per_floor = num_per_row * num_per_row
    for i in range(num_particles):
        floor = i // (num_per_floor) + 1 # prevent divided by zero
        row = (i % num_per_floor) // num_per_row
        col = (i % num_per_floor) % num_per_row
        positions[i] = ti.Vector([row*spacing, floor*spacing, col*spacing]) + init_pos


def T(a):
    if dim == 2:
        return a

    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5


init()
gui = ti.GUI('3D', background_color=0x112F41)
paused[None] = True
export_file=''
while gui.running and not gui.get_event(gui.ESCAPE):
    if gui.is_pressed(gui.SPACE):
        paused[None] = not paused[None]
    if not paused[None]:
        run_pbf()

    pos = positions.to_numpy()
    # export_file = r"PLY/res.ply"
    if export_file:
        writer = ti.tools.PLYWriter(num_vertices=num_particles)
        writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        writer.export_frame(gui.frame, export_file)
    gui.circles(T(pos), radius=1.5, color=0x66ccff)
    gui.show()
