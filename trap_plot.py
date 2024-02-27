import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def iso_trap_vertices(x, y, h, phi, psi,lambd):
    # Calculate half-width of the trapezoid's top base
    w_top = (1/h-lambd*l0*np.sin(phi/2))/2

    # Calculate half-width of the trapezoid's bottom base
    w_bottom = (1/h+lambd*l0*np.sin(phi/2))/2

    # Calculate x-coordinates of the vertices of the top base
    x_left_top = x - w_top
    x_right_top = x + w_top

    # Calculate y-coordinates of the vertices of the top and bottom bases
    y_top = y + h / 2
    y_bottom = y - h / 2

    # Rotate the vertices around the center
    rotation_matrix = np.array([
        [np.cos(-psi), np.sin(psi)],
        [np.sin(-psi), np.cos(-psi)]
    ])
    vertices = np.array([
        [x_left_top, y_top],   # Top left
        [x_right_top, y_top],  # Top right
        [x + w_bottom, y_bottom],  # Bottom right
        [x - w_bottom, y_bottom]   # Bottom left
    ])
    rotated_vertices = np.dot(vertices - [x, y], rotation_matrix) + [x, y]
    plt.gca().add_patch(Polygon(rotated_vertices, closed=True, fill=None, edgecolor='b'))

    return rotated_vertices




psi = np.pi/6
phi = 0
lambd = 1.1
l0 = 1
x_val = 0
y_val = 0

h = lambd*l0*np.cos(phi/2)
vertices = iso_trap_vertices(x_val,y_val,h,phi,psi,lambd)

# Plot the trapezoid
plt.gca().add_patch(Polygon(vertices, closed=True, fill=None, edgecolor='b'))
plt.title('Isosceles Trapezoid')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axis('equal')
plt.show()
