# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, debug=True, cpu_max_num_threads=1, advanced_optimization=False)

dim, n_grid, steps, dt = 3, 32, 25, 4e-4

num_particles = n_grid**dim // 2**(dim - 1)
dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
pbf_num_iters = 1
time_delta = 1.0 / 20.0
epsilon = 1e-5

h = 0.001
particle_radius = h * 1.05 #spacing


positions = ti.Vector.field(dim, float, num_particles)
old_positions = ti.Vector.field(dim, float, num_particles)
velocities = ti.Vector.field(dim, float, num_particles)

paused = ti.field(dtype=ti.i32, shape=())

@ti.func
def extern_force():
    for i in positions:
        g = ti.Vector([0., -0.1, 0.1])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)
    
@ti.func
def neighbor_search():
    pass

@ti.kernel
def prologue():
    for i in positions:
        old_positions[i] = positions[i]
    extern_force()
    neighbor_search()

@ti.func
def confine_position_to_boundary(p):
    bmin = 0. + 1e-3
    bmax = 2. - 1e-3
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax <= p[i]:
            p[i] = bmax - epsilon * ti.random()
    return p

@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta


@ti.kernel
def substep():
    pass

def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()


# @ti.kernel
# def init():
#     for i in range(num_particles):
#         positions[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.15


@ti.kernel
def init():
    cube_size = 0.4
    spacing = 0.02
    num_per_row = (int) (cube_size // spacing)
    num_per_floor = num_per_row * num_per_row

    for i in range(num_particles):
        floor = i // (num_per_floor) + 1 # prevent divided by zero
        row = (i % num_per_floor) // num_per_row
        col = (i % num_per_floor) % num_per_row

        positions[i] = ti.Vector([row*spacing, floor*spacing, col*spacing]) + 0.15

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

while gui.running and not gui.get_event(gui.ESCAPE):

    if gui.is_pressed(gui.SPACE):
        paused[None] = not paused[None]

    if not paused[None]:
    # if  gui.is_pressed('s'):
        run_pbf()


    pos = positions.to_numpy()

    export_file=''
    # export_file = r"PLY/res.ply"
    if export_file:
        writer = ti.tools.PLYWriter(num_vertices=num_particles)
        writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        writer.export_frame(gui.frame, export_file)
    gui.circles(T(pos), radius=1.5, color=0x66ccff)
    gui.show()
