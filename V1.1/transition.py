from main import *
from targeting import *


def GetCharacteristics(seed, print_characteristics=True):
    xspace = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    positions, extrapolated, separation, reattachment = GenerateStreamlinePoints(seed, xspace, min_y=cutoff_laminar)
    transition = DetermineTransitionPoint()[6]
    
    if print_characteristics: print("Separation point: ", separation[0], " | Reattachment point: ", reattachment[0], " | Transition point: ", transition)

    return separation, reattachment, transition

def DetermineTransitionPoint():
    # Get this: delta_max, delta_star, theta_star, delta_99, delta_95, H12, transition, transition_h12, transition_delta_star, thicknesses_in_heatmap = DetermineTransitionPoint()
    delta_max, delta_star, theta_star, delta_99, delta_95 = DetermineDeltaBoundaries()

    H12 = delta_star / theta_star
    transition_h12 = np.unique(X)[np.argmax(H12)]
    transition_delta_star = np.unique(X)[np.argmax(delta_star)]
    transition = transition_h12

    thicknesses_in_heatmap = {
    "δ_max": [True, "c-"],
    "δ*": [True, "m-"],
    "θ*": [True, "w-"],
    "δ_99": [True, "g-"],
    "δ_95": [True, "y-"],
    }

    thicknesses_in_heatmap["δ_max"].append(delta_max[:, 0])
    thicknesses_in_heatmap["δ_99"].append(delta_99)
    thicknesses_in_heatmap["δ_95"].append(delta_95)
    thicknesses_in_heatmap["δ*"].append(delta_star)
    thicknesses_in_heatmap["θ*"].append(theta_star)

    return delta_max, delta_star, theta_star, delta_99, delta_95, H12, transition, transition_h12, transition_delta_star, thicknesses_in_heatmap

def DetermineDeltaBoundaries():
    delta_max = np.array([[np.unique(Y)[i], i] for i in np.argmax(U_grid, axis=1)])  # point of maximum velocity along span
    
    delta_star = []
    delta_99 = []
    delta_95 = []
    theta_star = []

    for j, (dmax, i) in enumerate(delta_max):
        i = int(i)
        umax = U_grid[j, i]
        d99, d95 = 0, 0
    
        for k, uj in enumerate(U_grid[j, : i + 1]):
            if uj > 0.95 and d95 == 0: d95 = np.unique(Y)[k]
            elif uj > 0.99: 
                d99 = np.unique(Y)[k]
                break
    
        dstar = simpson([(1 - (uj / umax)) for uj in U_grid[j, : i + 1]], np.unique(Y)[: i + 1])
        thstar = simpson([(uj / umax) * (1 - (uj / umax)) for uj in U_grid[j, : i + 1]], np.unique(Y)[: i + 1],)

        delta_95.append(d95)
        delta_99.append(d99)
        delta_star.append(dstar)
        theta_star.append(thstar)

    delta_star, theta_star, delta_99, delta_95 = np.array(delta_star), np.array(theta_star), np.array(delta_99), np.array(delta_95),
    return delta_max, delta_star, theta_star, delta_99, delta_95

def PlotBoundaryLayerThickness(interpolate=False):
    delta_max, delta_star, theta_star, delta_99, delta_95, H12, transition, transition_h12, transition_delta_star, thicknesses_in_heatmap = DetermineTransitionPoint()
    fig_trnst, ax_trnst = plt.subplots()
    plt1 = ax_trnst.plot(np.unique(X), delta_max[:, 0], "k-", label="$\delta_{max}$")
    plt2 = ax_trnst.plot(np.unique(X), delta_star, "r-", label="$\delta{*}$")
    plt3 = ax_trnst.plot(np.unique(X), theta_star, "b-", label="$\\theta{*}$")
    plt4 = ax_trnst.plot(np.unique(X), delta_99, "g-", label="$\delta_{99}$")
    plt5 = ax_trnst.plot(np.unique(X), delta_95, "y-", label="$\delta_{95}$")
    ax_trnst.set_xlabel("x/c [-]")
    ax_trnst.set_ylabel("y/c [-]")
    ax_trnst.set(ylim=(0, ymax), xlim=(xmin, xmax))
    ax_trnst_2 = ax_trnst.twinx()
    plt6 = ax_trnst_2.plot(np.unique(X), H12, "m-", label="H$_{12}$")
    ax_trnst_2.set_ylabel("H$_{12}$ [-]")
    ax_trnst_2.set(ylim=(ymin, np.max(H12) * 1.03))

    lns = plt1 + plt2 + plt3 + plt4 + plt5 + plt6
    labs = [l.get_label() for l in lns]
    ax_trnst.legend(lns, labs, loc=0)

    plt.show()

