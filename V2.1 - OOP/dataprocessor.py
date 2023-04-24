import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
from scipy.interpolate import RegularGridInterpolator, interp2d
from scipy.integrate import solve_ivp, simpson

# from streamline import *


# Initialize some fixed variables
x_dim, y_dim = 395, 57
ylim = 57   
minl = (57 - ylim)*395
step = 20  


class DataProcessor():
    def __init__(self, data_path, data_resolution):
        self.path = data_path
        self.resolution = data_resolution
        self.acceleration_resolution = 5

        self.xcount, self.ycount = x_dim * self.resolution, y_dim * self.resolution

        self.storeData()
        self.estimateLaminarHeight(seperation_point_estimate=0.45)

        
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

        xh = np.linspace(self.X.min(), self.X.max(), self.xcount)
        yh = np.linspace(self.Y.min(), self.Y.max(), self.ycount)
        xy_grid = np.array([[x, y] for x in xh for y in yh])

        u_intpl, _ = self.getGridInterpolator()

        absolute_velocity = u_intpl(xy_grid).reshape((self.xcount,self.ycount)).T
        return absolute_velocity
    
    def _getRawVelocityOverGrid(self):

        xh = np.linspace(self.X.min(), self.X.max(), x_dim)
        yh = np.linspace(self.Y.min(), self.Y.max(), y_dim)
        xy_grid = np.array([[x, y] for x in xh for y in yh])

        u_intpl, _ = self.getGridInterpolator()

        absolute_velocity = u_intpl(xy_grid).reshape((x_dim, y_dim)).T
        return absolute_velocity


    def getAccelerationOverGrid(self):
        absolute_velocity = self._getRawVelocityOverGrid()
        absolute_acceleration = self._FiniteDifferenceMethod(absolute_velocity, dx=1e-3, stencil=5, output="abs")
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


    def plot(self, plot_type="velocity heatmap"):
        """plot_type options: "velocity heatmap", "acceleration heatmap" """
        
        if (plot_type == "velocity heatmap"):
            heatmap_grid = self.getVelocityOverGrid()
            self._plotVelocityHeatmap(heatmap_grid)

        elif (plot_type == "acceleration heatmap"):
            heatmap_grid = self.getAccelerationOverGrid()
            self._plotAccelerationHeatmap(heatmap_grid)
        
        plt.show()

    def _plotVelocityHeatmap(self, heatmap_grid):
        fig_heatmap, ax_heatmap = plt.subplots()
        heatmap = ax_heatmap.imshow(heatmap_grid[::-1,:], extent=(self.raw_data[-1,0], self.raw_data[0,0], self.raw_data[-1,1], self.raw_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
        plt.colorbar(heatmap, label="Absolute velocity", ax=ax_heatmap)
        ax_heatmap.set_xlabel("x/c [-]")
        ax_heatmap.set_ylabel("y/c [-]")

    def _plotAccelerationHeatmap(self, heatmap_grid):
        fig_gradient, ax_gradient = plt.subplots()
        heatmap = ax_gradient.imshow(heatmap_grid[::-1,:], extent=(self.raw_data[-1,0], self.raw_data[0,0], self.raw_data[-1,1], self.raw_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
        plt.colorbar(heatmap, label="Absolute acceleration", ax=ax_gradient)
        ax_gradient.set_xlabel("x/c [-]")
        ax_gradient.set_ylabel("y/c [-]")


    def estimateLaminarHeight(self, seperation_point_estimate, n_steps=10, return_=False):
        uCi, _ = self._getVelocityComponents()

        umax, umin = uCi.max(), uCi.min()

        # TODO: Find auto mean function
        mean = 0.55 

        contour_x, contour_y = self._getVelocityContourPoints(mean)

        heights = []
        for i in range(0, len(contour_x), int(len(contour_x/n_steps))):
            if contour_x[i] < seperation_point_estimate:
                heights.append(contour_y[i])
        estimated_height = np.mean(heights)

        self.laminar_height = estimated_height

        if return_: return estimated_height

    def _getVelocityComponents(self):
        u_intpl, v_intpl = self.getGridInterpolator()
        
        xi = np.linspace(self.X.min(), self.X.max(), x_dim)
        yi = np.linspace(self.Y.min(), self.Y.max(), y_dim)
        xy_points = np.array([[x,y] for x in xi for y in yi])

        uCi = u_intpl(xy_points)
        vCi = v_intpl(xy_points)

        return uCi, vCi
    
    def _getVelocityContourPoints(self, mean):

        # Create the grid for the matplotlib plot
        xh = np.linspace(self.X.min(), self.X.max(), self.xcount)
        yh = np.linspace(self.Y.min(), self.Y.max(), self.ycount)
        absolute_velocity = self.getVelocityOverGrid()

        # Let matplotlib calculate contour points and extract those without plotting anything
        fig_contour, ax_contour = plt.subplots()
        contour = ax_contour.contour(xh, yh, absolute_velocity.reshape(self.ycount, self.xcount), levels=[mean], colors="black", linewidths=1)
        p = contour.allsegs[0]
        plt.close(fig_contour)

        # Extract the contour points
        contour_x = p[0][:, 0]
        contour_y = p[0][:, 1]

        return contour_x, contour_y
    

    def getTargetContourVelocity(self, n_steps=15, degree=2, plot_hm=False):
        uCi, _ = self._getVelocityComponents()
        umax, umin = uCi.max(), uCi.min()
        mean = 0.5 * (umax + umin)

        # Retrieve the contour points set on that mean
        contour_x, contour_y = self._getContourPoints(mean)

        # Set automatic target contour boundaries based on the laminar height and existing contour points
        contour_cutoff_min = contour_x[np.where(contour_y > self.laminar_height*3)[0][0]]
        contour_cutoff_max = contour_x[np.argmax(contour_y)-300]

        points = []
        for i in range(len(contour_x)):
            if contour_cutoff_min < contour_x[i] < contour_cutoff_max and contour_y[i] > self.laminar_height:
                coord = [contour_x[i], contour_y[i]]
                points.append(coord)
        points = np.array(points)


        # Let's filter some of the selected points out
        filtered_points = []
        filter_step = int(len(points)/n_steps)

        for i in range(0, len(points), filter_step):
            filtered_points.append(points[i])
        filtered_points = np.array(filtered_points)

        # Using the contour points, fit a polynomial!
        coeff = np.polyfit(filtered_points[:, 0], filtered_points[:, 1], degree)
        polynomial = np.poly1d(coeff)

        if plot_hm:
            heatmap_grid = self.getVelocityOverGrid()
            fig_heatmap, ax_heatmap = plt.subplots()
            heatmap = ax_heatmap.imshow(heatmap_grid[::-1,:], extent=(self.raw_data[-1,0], self.raw_data[0,0], self.raw_data[-1,1], self.raw_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
            plt.colorbar(heatmap, label="Absolute velocity", ax=ax_heatmap)
            ax_heatmap.set_xlabel("x/c [-]")
            ax_heatmap.set_ylabel("y/c [-]")

            ax_heatmap.scatter(filtered_points[:,0], filtered_points[:,1], color="black", s=1)
            ax_heatmap.plot(filtered_points[:,0], polynomial(filtered_points[:,0]), color="red", linewidth=0.8)

            plt.show()

        return filtered_points, polynomial
    
    def getTargetContourAcceleration(self, degree=2, plot=False):

        raw_acceleration_grid = self.getAccelerationOverGrid()
        a_interpolator = self._getAccelerationInterpolator(raw_acceleration_grid)

        xi = np.linspace(self.X.min(), self.X.max(), self.xcount*self.acceleration_resolution)
        yi = np.linspace(self.Y.min(), self.Y.max(), self.ycount*self.acceleration_resolution)
        xy_points = np.array([[x,y] for x in xi for y in yi])

        interpolated_acceleration = []
        for point in xy_points:
            acc = a_interpolator(point[0], point[1])
            interpolated_acceleration.append(acc)

        interpolated_acceleration = np.array(interpolated_acceleration).reshape((self.xcount*self.acceleration_resolution, self.ycount*self.acceleration_resolution))

        '''
        Steps to process:
        1. Cut off the lower max acceleration bar (-> "Laminar height" part + little extra as safety)
        2. Find max acc. -> contour that shitßßß
        3. From that contour filter out some points and create polynomial
        '''

        # Find maximum acceleration -> y location of point where this occurs
        
        amax = np.argmax(interpolated_acceleration)
        point = xy_points[amax] + 1e-3
        # print(point)

        xy_points = xy_points.reshape((self.xcount*self.acceleration_resolution, self.ycount*self.acceleration_resolution, 2))
        index = np.where(xy_points[:,:,1] <= point[1])
        interpolated_acceleration[index] = 0.0 

        xc_min, xc_max = 0.51, 0.64 #TODO: Automate this!

        xspace = np.arange(xc_min, xc_max, 1e-4)
        y_min, y_max = point[1], 0.03
        yspace = np.arange(y_min, y_max, 1e-5)

        contour_points = []
        for x in xspace:
            accelerations = []
            for y in yspace:
                acc = a_interpolator(x, y)
                accelerations.append(acc)
            accelerations = np.array(accelerations)
            
            max_index = np.argmax(accelerations)
            contour_points.append([x, yspace[max_index]])
        contour_points = np.array(contour_points)
        
        # Now that we have our contour points, let's find the best fitting polynomial for it and export that shit
        coeff = np.polyfit(contour_points[:, 0], contour_points[:, 1], degree)
        polynomial = np.poly1d(coeff)


        interpolated_acceleration = interpolated_acceleration.T
        if plot:
            fig_gradient, ax_gradient = plt.subplots()
            heatmap = ax_gradient.imshow(interpolated_acceleration[::-1,:], extent=(self.raw_data[-1,0], self.raw_data[0,0], self.raw_data[-1,1], self.raw_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
            plt.colorbar(heatmap, label="Absolute acceleration [1/U$_{inf}$]", ax=ax_gradient)
            ax_gradient.set_xlabel("x/c [-]")
            ax_gradient.set_ylabel("y/c [-]")
            
            # ax_gradient.scatter(contour_points[:,0], contour_points[:,1], color="black", s=1)
            ax_gradient.plot(contour_points[:,0], polynomial(contour_points[:,0]), color="black", linewidth=1.2)

            plt.show()

        return contour_points, polynomial

    def _getAccelerationInterpolator(self, acceleration_grid):
        acc_xcount, acc_ycount = self.xcount*self.acceleration_resolution, self.ycount*self.acceleration_resolution
        interpolator = interp2d(np.unique(self.X), np.unique(self.Y), acceleration_grid)    

        return interpolator
