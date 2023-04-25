import numpy as np
import matplotlib.pyplot as plt
from dataprocessor import DataProcessor

from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp, simpson

from dataprocessor import *

cutoff_reattachment = 0.007
cutoff_seperation = 0.0008
cutoff_laminar = 0.007
cutoff_turbulent = 0.0007

laminar_height = 0.001 
xc_cutoff = 0.65
xc_max = 0.52


class Streamline():

    def __init__(self, dataprocessor: DataProcessor, seed=np.array([0.47,0.04])):
        self.seed = np.array(seed)
        self.dataprocessor = dataprocessor
        self.resolution = self.dataprocessor.resolution

        self.xspace = np.linspace(self.dataprocessor.X.min(), self.dataprocessor.X.max(), 395*self.resolution)

        self.generateStreamlinePoints(self.xspace, min_y=cutoff_laminar)
    
    def f(self, t, xy):
        if xy[1] < self.dataprocessor.y.min() or xy[1] > self.dataprocessor.y.max() or xy[0] < self.dataprocessor.x.min() or xy[0] > self.dataprocessor.x.max():
            return np.array([0, 0])
        u_intpl, v_intpl = self.dataprocessor.getGridInterpolator()
        return np.squeeze([u_intpl(xy), v_intpl(xy)])
    
    def generateStreamlinePoints(self, xspace, min_y):
        solution = solve_ivp(self.f, [0, 10], self.seed, first_step=1e-3, max_step=1e-2, method="RK45", dense_output=True)
        self.positions = solution.y
        while (self.positions[0, -1] > self.dataprocessor.x.max() or self.positions[0, -1] < self.dataprocessor.x.min() or self.positions[1, -1] > self.dataprocessor.y.max() or self.positions[1, -1] < min_y):
            self.positions = self.positions[:,:-1]
        
        self.position_intpl = interp1d(self.positions[0], self.positions[1], kind="linear", fill_value="extrapolate")
        
        self.extrapolated = np.array([[xk, self.position_intpl(xk)] for xk in xspace if self.position_intpl(xk) > self.dataprocessor.y.min()])
        self.separation, self.reattachment = self.extrapolated[0,:], self.extrapolated[-1,:]


    def retrieveCleanPositions(self):
        # Clean up the streamline positions (instead of [(x1, x2), (y1, y2)] -> [(x1,y1), (x2, y2)])
        x_points, y_points = self.positions[0], self.positions[1]
        xy_points = [[xs, ys] for xs in x_points for ys in y_points]
        positions = np.array(xy_points)
        return positions

    def plot(self, plot_type="streamline"):
        """plot_type options: "streamline", "boundary layer thicknesses"""
        if plot_type == "streamline":
            heatmap_grid = self.dataprocessor.getVelocityOverGrid()
            
            # Plot the heatmap
            fig_heatmap, ax_heatmap = plt.subplots()
            heatmap = ax_heatmap.imshow(heatmap_grid[::-1,:], extent=(self.dataprocessor.raw_data[-1,0], self.dataprocessor.raw_data[0,0], self.dataprocessor.raw_data[-1,1], self.dataprocessor.raw_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
            plt.colorbar(heatmap, label="Absolute velocity", ax=ax_heatmap)
            ax_heatmap.set_xlabel("x/c [-]")
            ax_heatmap.set_ylabel("y/c [-]")

            # Plot the cutoff lines
            ax_heatmap.plot((self.dataprocessor.x.min(), self.dataprocessor.x.max()), (cutoff_reattachment, cutoff_reattachment), "k--", linewidth=0.5)
            ax_heatmap.plot((self.dataprocessor.x.min(), self.dataprocessor.x.max()), (laminar_height, laminar_height), "k--", linewidth=0.5) 
            
            ax_heatmap.plot(self.extrapolated[:,0], self.extrapolated[:,1], "r--")
            ax_heatmap.plot(self.positions[0], self.positions[1], "r-")
            ax_heatmap.plot((self.separation[0], self.reattachment[0]), (self.separation[1], self.reattachment[1]), "ro")
            
            # TODO: ADD Transition point determination!

        elif plot_type == "boundary layer thicknesses":
            pass # TODO: Implement function

        plt.show()

    def retrieveCharacteristics(self, decimals=4):
        transition = np.round(self.retrieveTransitionPoint(),decimals=decimals)
        seperation = np.round(self.separation[0], decimals=decimals)
        reattachment = np.round(self.reattachment[0], decimals=decimals)
        return seperation, reattachment, transition
    
    def determineTransitionPoint(self):
        # Get this: delta_max, delta_star, theta_star, delta_99, delta_95, H12, transition, transition_h12, transition_delta_star, thicknesses_in_heatmap = DetermineTransitionPoint()
        delta_max, delta_star, theta_star, delta_99, delta_95 = self.determineDeltaBoundaries()

        H12 = delta_star / theta_star
        transition_h12 = np.unique(self.dataprocessor.X)[np.argmax(H12)]
        transition_delta_star = np.unique(self.dataprocessor.X)[np.argmax(delta_star)]
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
    
    def determineDeltaBoundaries(self):
        delta_max = np.array([[np.unique(self.dataprocessor.Y)[i], i] for i in np.argmax(self.dataprocessor.U_grid, axis=1)])  # point of maximum velocity along span
    
        delta_star = []
        delta_99 = []
        delta_95 = []
        theta_star = []

        for j, (dmax, i) in enumerate(delta_max):
            i = int(i)
            umax = self.dataprocessor.U_grid[j, i]
            d99, d95 = 0, 0
        
            for k, uj in enumerate(self.dataprocessor.U_grid[j, : i + 1]):
                if uj > 0.95 and d95 == 0: d95 = np.unique(self.dataprocessor.Y)[k]
                elif uj > 0.99: 
                    d99 = np.unique(self.dataprocessor.Y)[k]
                    break
        
            dstar = simpson([(1 - (uj / umax)) for uj in self.dataprocessor.U_grid[j, : i + 1]], np.unique(self.dataprocessor.Y)[: i + 1])
            thstar = simpson([(uj / umax) * (1 - (uj / umax)) for uj in self.dataprocessor.U_grid[j, : i + 1]], np.unique(self.dataprocessor.Y)[: i + 1],)

            delta_95.append(d95)
            delta_99.append(d99)
            delta_star.append(dstar)
            theta_star.append(thstar)

        delta_star, theta_star, delta_99, delta_95 = np.array(delta_star), np.array(theta_star), np.array(delta_99), np.array(delta_95),
        return delta_max, delta_star, theta_star, delta_99, delta_95
    
    def retrieveTransitionPoint(self):
        transition = self.determineTransitionPoint()[6]
        self.transition = transition
        return transition
    
    def plotBoundaryLayerThickness(self, interpolate=False):
        xmin, xmax = self.dataprocessor.x.min()/1.02, self.dataprocessor.x.max()*1.02
        ymin, ymax = -0.001, self.dataprocessor.y.max()*1.03

        delta_max, delta_star, theta_star, delta_99, delta_95, H12, transition, transition_h12, transition_delta_star, thicknesses_in_heatmap = self.determineTransitionPoint()
        fig_trnst, ax_trnst = plt.subplots()
        plt1 = ax_trnst.plot(np.unique(self.dataprocessor.X), delta_max[:, 0], "k-", label="$\delta_{max}$")
        plt2 = ax_trnst.plot(np.unique(self.dataprocessor.X), delta_star, "r-", label="$\delta{*}$")
        plt3 = ax_trnst.plot(np.unique(self.dataprocessor.X), theta_star, "b-", label="$\\theta{*}$")
        plt4 = ax_trnst.plot(np.unique(self.dataprocessor.X), delta_99, "g-", label="$\delta_{99}$")
        plt5 = ax_trnst.plot(np.unique(self.dataprocessor.X), delta_95, "y-", label="$\delta_{95}$")
        ax_trnst.set_xlabel("x/c [-]")
        ax_trnst.set_ylabel("y/c [-]")
        ax_trnst.set(ylim=(0, ymax), xlim=(xmin, xmax))
        ax_trnst_2 = ax_trnst.twinx()
        plt6 = ax_trnst_2.plot(np.unique(self.dataprocessor.X), H12, "m-", label="H$_{12}$")
        ax_trnst_2.set_ylabel("H$_{12}$ [-]")
        ax_trnst_2.set(ylim=(ymin, np.max(H12) * 1.03))

        lns = plt1 + plt2 + plt3 + plt4 + plt5 + plt6
        labs = [l.get_label() for l in lns]
        ax_trnst.legend(lns, labs, loc=0)

        plt.show()
