import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm
from matplotlib.widgets import Slider
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp, simpson
import os
import random


# Import the tutor's data
piv_file_path = os.path.join(os.path.dirname(__file__), "./PIV.dat")
piv_data = np.genfromtxt(piv_file_path, skip_header=1, delimiter=",")

# Data settings
xcount, ycount = 395, 57
ylim = 57

# Package related settings
vccsm = cm.turbo
hmcm = cm.jet

# Change-able variables
laminar_height = 0.001      # Estimated (laminar) boundary thickness
step = 20                   # 
initial_seeds = np.array([[0.48, laminar_height],[0.59, laminar_height]]) # Array of seeds (base)
heatmap_resolution = 15     # Resolution of the heatmap gradient plot
xc_cutoff = 0.65            # point at which to cutoff the streamline analysis
xc_max = 0.52               # maximum xc for valid streamline (eye-balling)
cutoff_laminar = 0.007      # same as laminar height, but in diego's work (also for turbulent below) 
cutoff_turbulent = 0.0007


cutoff_seperation = 0.0008 #<Guess
cutoff_reattachment = 0.007 #<Guess

# ------------------ pre-processing the imported data ------------------
minl = (57 - ylim)*395
x, y = piv_data[:, 0], piv_data[:, 1]
xmin, xmax = x.min()/1.02, x.max()*1.02
ymin, ymax = -0.001, y.max()*1.03
u, v = piv_data[:, 2], piv_data[:, 3]
X, Y = piv_data[minl::step, 0], piv_data[minl::step, 1]
U, V = piv_data[minl::step, 2], piv_data[minl::step, 3]

colors = np.linalg.norm(np.column_stack((U, V)), axis=1)
norm = Normalize()
norm.autoscale(colors)

# Sort the position and velocity data
sorted_data = piv_data[np.lexsort((piv_data[:,1], piv_data[:,0]))]
X, Y = sorted_data[:, 0], sorted_data[:, 1]
U, V = sorted_data[:, 2], sorted_data[:, 3]

# Create a regularly spaced position grid spanning the domain of x and y 
xi = np.linspace(X.min(), X.max(), xcount)
yi = np.linspace(Y.min(), Y.max(), ycount)
xy_points = np.array([[x, y] for x in xi for y in yi])

# Create a regularly spaced velocity grid spanning the domain of x and y 
U_grid = U.reshape((xcount, ycount))
V_grid = V.reshape((xcount, ycount))

# Bicubic interpolation
u_intpl = RegularGridInterpolator((np.unique(X), np.unique(Y)), U_grid)
v_intpl = RegularGridInterpolator((np.unique(X), np.unique(Y)), V_grid)
uCi, vCi = u_intpl(xy_points), v_intpl(xy_points)
speed = np.sqrt(uCi**2 + vCi**2)

norm = Normalize()
norm.autoscale(speed)

# Function : Plot vector field given X and Y and the corresponding velocities U and V
def PlotVectorField():
    fig_piv, ax_piv = plt.subplots()
    ax_piv.quiver(X, Y, U, V, color=vccsm(norm(colors)), pivot="mid", scale=100, scale_units="xy", width=0.001, headwidth=3, headlength=4, headaxislength=3)
    ax_piv.set_xlabel("x/c [-]")
    ax_piv.set_ylabel("y/c [-]")
    ax_piv.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_piv, label="Absolute velocity [1/U$_{inf}$]")
    plt.show()

# Function : Plot heatmap given seed, and settings
def PlotVelocityHeatmap(seed):
    global hmcm
    norm = "linear"

    # Create a grid for the heatmap (sizing it up on resolution)
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    # Use the RegularGridInterpolator to find the nicely interpolated grid
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T

    # Create a discerete colormap
    discrete_colormap = False    
    if discrete_colormap:
        cmaplist = [hmcm(i) for i in range(hmcm.N)]
        hmcm = LinearSegmentedColormap.from_list("Custom Discrete map", cmaplist, hmcm.N)
        bounds = np.linspace(absv.min(), absv.max(), 21)
        norm = BoundaryNorm(bounds, hmcm.N)

    # Create the subplots for the heatmap and colorbar
    fig_heatmap, ax_heatmap = plt.subplots()
    heatmap = ax_heatmap.imshow(absv[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
    plt.colorbar(heatmap, label="Absolute velocity [1/U$_{inf}$]", ax=ax_heatmap)
    ax_heatmap.set_xlabel("x/c [-]")
    ax_heatmap.set_ylabel("y/c [-]")

    # If a seed is filled into the function, then add it to the heatmap :)
    ax_heatmap.plot((x.min(), x.max()), (cutoff_reattachment, cutoff_reattachment), "k--", linewidth=0.5)
    ax_heatmap.plot((x.min(), x.max()), (laminar_height, laminar_height), "k--", linewidth=0.5) 
        
    sol, extrapolated, separation, reattachment = GenerateStreamlinePoints(seed, xh, min_y=cutoff_laminar)
    print("Separation point: ", separation[0])
    print("Reattachment point: ", reattachment[0])

    # Plot extrapolated sections (laminar and turbulent section)
    ax_heatmap.plot(extrapolated[:,0], extrapolated[:,1], "r--")
    ax_heatmap.plot(sol[0], sol[1], "r-")
    ax_heatmap.plot((separation[0], reattachment[0]), (separation[1], reattachment[1]), "ro")
    plt.show()

# Function : Streamline function
def f(t, xy):
    if xy[1] < y.min() or xy[1] > y.max() or xy[0] < x.min() or xy[0] > x.max():
        return np.array([0, 0])
    return np.squeeze([u_intpl(xy), v_intpl(xy)])

# Function : Find the streamline given a starting(/seed) point
def GenerateStreamline(seed_point, min_y=y.min()): #get all the points with meh form
    solution = solve_ivp(f, [0, 10], seed_point, first_step=1e-3, max_step=1e-2, method="RK45", dense_output=True)
    positions = solution.y
    while (positions[0, -1] > x.max() or positions[0, -1] < x.min() or positions[1, -1] > y.max() or positions[1, -1] < min_y):
        positions = positions[:,:-1]
    intpl = interp1d(positions[0], positions[1], kind="linear", fill_value="extrapolate")
    return positions, intpl
    # --> if you get an error here, the seed you selected is out of bounds! <x.min() OR >x.max()

def GenerateStreamlinePoints(seed, xspace, min_y=y.min()): #get the points over a certain xspace
    positions, intpl = GenerateStreamline(seed, min_y)
    extrapolated = np.array([[xk, intpl(xk)] for xk in xspace if intpl(xk) > y.min()])
    separation, reattachment = extrapolated[0,:], extrapolated[-1,:]
    return positions, extrapolated, separation, reattachment

def GetCleanStreamlinePositions(seed, xspace): #get all the points available in an easy to access numpy array
    streamline_positions = GenerateStreamlinePoints(seed, xspace, min_y=cutoff_reattachment)[0]
    x_points, y_points = streamline_positions[0], streamline_positions[1]
    xy_points = [[xs, ys] for xs in x_points for ys in y_points]
    return np.array(xy_points)



# ------------------------------------------------