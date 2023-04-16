import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp, simpson


# Initialize some fixed variables
x_dim, y_dim = 395, 57
ylim = 57   
minl = (57 - ylim)*395
step = 20  


class DataProcessor():
    def __init__(self, data_path, data_resolution):
        self.path = data_path
        self.resolution = data_resolution

        self.storeData()
        
    def storeData(self):
        piv_file_path = os.path.join(os.path.dirname(__file__), "../PIV.dat")
        piv_data = np.genfromtxt(piv_file_path, skip_header=1, delimiter=",")
        self.raw_data = piv_data

        self.x, self.y = self.raw_data[:, 0], self.raw_data[:, 1]
        xmin, xmax = self.x.min()/1.02, self.x.max()*1.02
        ymin, ymax = -0.001, self.y.max()*1.03
        self.u, self.v = self.raw_data[:, 2], self.raw_data[:, 3]
        self.X, self.Y = self.raw_data[minl::step, 0], self.raw_data[minl::step, 1]
        self.U, self.V = self.raw_data[minl::step, 2], self.raw_data[minl::step, 3]

        sorted_data = self.raw_data[np.lexsort((self.raw_data[:,1], self.raw_data[:,0]))]
        self.X, self.Y = sorted_data[:, 0], sorted_data[:, 1]
        self.U, self.V = sorted_data[:, 2], sorted_data[:, 3]
        
    def getGridInterpolator(self):
        xi = np.linspace(self.X.min(), self.X.max(), x_dim)
        yi = np.linspace(self.Y.min(), self.Y.max(), y_dim)
        xy_points = np.array([[x, y] for x in xi for y in yi])

        U_grid = self.U.reshape((x_dim, y_dim))
        V_grid = self.V.reshape((x_dim, y_dim))

        u_intpl = RegularGridInterpolator((np.unique(self.X), np.unique(self.Y)), U_grid)
        v_intpl = RegularGridInterpolator((np.unique(self.X), np.unique(self.Y)), V_grid)
        
        return u_intpl, v_intpl
    
    def getVelocityOverGrid(self):
        xcount, ycount = x_dim * self.resolution, y_dim * self.resolution

        xh = np.linspace(self.X.min(), self.X.max(), xcount)
        yh = np.linspace(self.Y.min(), self.Y.max(), ycount)
        xy_grid = np.array([[x, y] for x in xh for y in yh])

        u_intpl, v_intpl = self.getGridInterpolator()

        absolute_velocity = u_intpl(xy_grid).reshape((xcount,ycount)).T
        return absolute_velocity
    

    def getAccelerationOverGrid(self):
        absolute_velocity = self.getVelocityOverGrid()
        absolute_acceleration = self._FiniteDifferenceMethod(absolute_velocity, dx=0.1, stencil=5, output="abs")

        return absolute_acceleration

    def _FiniteDifferenceMethod(self, v, dx=1e-5, stencil=5, output="abs"):
        nx, ny = v.shape

        # Initialize arrays to hold the accelerations in the x and y directions
        ax = np.zeros((nx, ny))
        ay = np.zeros((nx, ny))
        acceleration = np.zeros((nx, ny))

        # Compute the accelerations using the Finite Difference Method - Higher Order
        if stencil == 5:
            for i in range(2, nx-2):
                for j in range(2, ny-2):
                    ax[i,j] = (-v[i,j+2] + 8*v[i,j+1] - 8*v[i,j-1] + v[i,j-2]) / (12*dx)
                    ay[i,j] = (-v[i+2,j] + 8*v[i+1,j] - 8*v[i-1,j] + v[i-2,j]) / (12*dx)
                    acceleration[i, j] = np.sqrt(ax[i,j]**2 + ay[i,j]**2)

        elif stencil == 6:
            for i in range(3, nx-3):
                for j in range(3, ny-3):
                    ax[i,j] = (-v[i,j+3] + 9*v[i,j+2] - 45*v[i,j+1] + 45*v[i,j-1] - 9*v[i,j-2] + v[i,j-3]) / (60*dx)
                    ay[i,j] = (-v[i+3,j] + 9*v[i+2,j] - 45*v[i+1,j] + 45*v[i-1,j] - 9*v[i-2,j] + v[i-3,j]) / (60*dx)
                    acceleration[i, j] = np.sqrt(ax[i,j]**2 + ay[i,j]**2)
        
        else: print("Finite Difference Method : Incorrect point-stencil chosen (5,6) are available")

        if output == "abs": 
            return acceleration
        elif output == "sep":
            return ax, ay
        else:
            print("Finite Difference Method : Incorrect output method for the Finite Difference Method")


    def plot(self, plot_type="velocity heatmap", hold_show=False):
        """plot_type options: "velocity heatmap", "acceleration heatmap" """
        
        if (plot_type == "velocity heatmap"):
            heatmap_grid = self.getVelocityOverGrid()
            self._plotVelocityHeatmap(heatmap_grid, hold_show)

        elif (plot_type == "acceleration heatmap"):
            heatmap_grid = self.getAccelerationOverGrid()
            self._plotAccelerationHeatmap(heatmap_grid, hold_show)


        if not hold_show:  
            plt.show()

    def _plotVelocityHeatmap(self, heatmap_grid, hold_show):
        fig_heatmap, ax_heatmap = plt.subplots()
        heatmap = ax_heatmap.imshow(heatmap_grid[::-1,:], extent=(self.raw_data[-1,0], self.raw_data[0,0], self.raw_data[-1,1], self.raw_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
        plt.colorbar(heatmap, label="Absolute velocity", ax=ax_heatmap)
        ax_heatmap.set_xlabel("x/c [-]")
        ax_heatmap.set_ylabel("y/c [-]")

        if hold_show: return ax_heatmap

    def _plotAccelerationHeatmap(self, heatmap_grid, hold_show):
        fig_gradient, ax_gradient = plt.subplots()
        heatmap = ax_gradient.imshow(heatmap_grid[::-1,:], extent=(self.raw_data[-1,0], self.raw_data[0,0], self.raw_data[-1,1], self.raw_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
        plt.colorbar(heatmap, label="Absolute acceleration", ax=ax_gradient)
        ax_gradient.set_xlabel("x/c [-]")
        ax_gradient.set_ylabel("y/c [-]")

        if hold_show: return ax_gradient

        


