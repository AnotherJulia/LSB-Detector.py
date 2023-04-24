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

        xspace = np.linspace(self.dataprocessor.X.min(), self.dataprocessor.X.max(), 395*self.resolution)
        self.generateStreamlinePoints(xspace, min_y=cutoff_laminar)
    
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