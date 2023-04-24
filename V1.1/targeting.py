from main import *

# ----------------------------------------------------------------
#       File goal: Get the targeting and fitting correct!
# ----------------------------------------------------------------

def RunTargeting(n_generations=3, print_progress=False, target_method="mean"):
    laminar_height = EstimateLaminarHeight()
    seed = GenerationalFitting(n_generations=n_generations, print_progress=print_progress, target_method=target_method)
    #PlotTargetHeatmap(seed, target_contour=None, target_polynomial=None)
    return seed

# --------------------------------------------------------------------

def EstimateLaminarHeight(n_steps=10): #Function : Estimate the laminar height of the functionx
    umax, umin = uCi.max(), uCi.min()
    # Find mean of the velocity field = color of dividing streamline in gradient plot
    mean = 0.55
    contour = [mean]

    # Generate grid
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T
    
    # Calculate contour points, without plotting the contour map
    fig_contour, ax_contour = plt.subplots()
    contour = ax_contour.contour(xh, yh, absv.reshape(ycount*heatmap_resolution,xcount*heatmap_resolution), levels=contour, colors="black", linewidths=1)
    p = contour.allsegs[0] # extract contour points
    plt.close(fig_contour)


    # Extract contour points in the right format
    contour_x = p[0][:, 0]
    contour_y = p[0][:, 1]

    estimated_seperation_point = 0.45 # TODO: AUTOMATE THIS

    # then instead of looking beyond xc_cutoff, we want to find the height of the mean around 0.0-0.1
    heights=[]
    for i in range(0, len(contour_x), int(len(contour_x)/n_steps)):
        if contour_x[i]<estimated_seperation_point:
            heights.append(contour_y[i])
    mean_heights = np.mean(heights)

    #print("Laminar Height: ", mean_heights)

    return mean_heights

# TODO: ADD MORE ACCURATE TARGET METHOD
def GetTargetContour(n_steps=15, degree=6, min_y=laminar_height, plot_hm=False): # Function : Find target polynomial using basic contour + mean
    # => using interpolation of points to find target -> then find "closest" streamline
    
    umax, umin = uCi.max(), uCi.min()
    mean = 0.5*(umax + umin)
    #print("Contour Height Mean: ", mean)
    contour = [mean]

    # Generate grid
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T
    
    # Plot the contour corresponding to the "temperature" on the heatmap of contour=[mean]
    fig_contour, ax_contour = plt.subplots()
    contour = ax_contour.contour(xh, yh, absv.reshape(ycount*heatmap_resolution,xcount*heatmap_resolution), levels=contour, colors="black", linewidths=1)
    p = contour.allsegs[0] # extract points from contour map
    plt.close(fig_contour) #close the contour map to avoid it showing

    # Extract points from the contour map in the right format
    contour_x = p[0][:, 0]
    contour_y = p[0][:, 1]

    # Set automatic target contour boundaries based on the laminar height and existing contour points
    contour_cutoff_min = contour_x[np.where(contour_y > laminar_height*3)[0][0]]
    contour_cutoff_max = contour_x[np.argmax(contour_y)-300]

    points = []
    # Check if points are within region
    for i in range(len(contour_x)):
        if contour_cutoff_min < contour_x[i] < contour_cutoff_max and contour_y[i] > min_y: # check if the contour falls within region 
            coord = [contour_x[i], contour_y[i]]
            points.append(coord)
    points = np.array(points)

    # Filter out some of the points
    contour_points = []
    filter_step = int(len(points)/n_steps)
    for i in range(0, len(points), filter_step):
        contour_points.append(points[i])
    contour_points = np.array(contour_points)

    # From the contour points, automatically find the polynomial / continuous contour
    coeff = np.polyfit(contour_points[:, 0], contour_points[:,1], degree)
    polynomial = np.poly1d(coeff)

    # Plot the target contour and contour points
    if plot_hm:
        fig_hm, ax_hm = plt.subplots()
        
        heatmap = ax_hm.imshow(absv[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
        plt.colorbar(heatmap, label="Absolute velocity [1/U$_{inf}$]", ax=ax_hm)
        ax_hm.set_xlabel("x/c [-]")
        ax_hm.set_ylabel("y/c [-]")

        ax_hm.scatter(contour_points[:,0], contour_points[:,1], color="black", s=1) #add contour points
        ax_hm.plot(contour_points[:,0], polynomial(contour_points[:,0]), color="red", linewidth=0.8) #add the line in between the contour points

        plt.show()

    return contour_points, polynomial

def GetTargetChange(boundary_factor=1.0, plot=False):
    acc = GetAccelerationMap(output="abs")

    acc_resolution = 1

    acc_grid = acc.reshape((xcount, ycount))
    acc_interpolator = RegularGridInterpolator((np.unique(X), np.unique(Y)), acc_grid)

    xh = np.arange(X.min(), X.max(), xcount*acc_resolution)
    yh = np.arange(Y.min(), Y.max(), ycount*acc_resolution)
    xy_points = np.array([[xi,yi] for xi in xh for yi in yh])

    acc_interpolated = acc_interpolator(xy_points)
    print(acc_interpolated)


    # Plot the shitshow to test
    if plot:
        fig_gradient, ax_gradient = plt.subplots()
        heatmap = ax_gradient.imshow(acc_interpolated[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
        plt.colorbar(heatmap, label="Absolute acceleration [1/U$_{inf}$]", ax=ax_gradient)
        ax_gradient.set_xlabel("x/c [-]")
        ax_gradient.set_ylabel("y/c [-]")


    plt.show()

def GetAccelerationMap(output="abs"):
    # Find the absolute velocity grid/map
    xh = np.linspace(X.min(), X.max(), xcount)
    yh = np.linspace(Y.min(), Y.max(), ycount)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    
    # Use the RegularGridInterpolator to find the nicely interpolated grid
    absv = u_intpl(xy_grid).reshape((xcount,ycount)).T   

    # Find the actual acceleration map using the finite differenec method
    print("Finding acceleraton map")
    acceleration = FiniteDifferenceMethod(absv, dx=0.1, stencil=5, output=output)

    # remove the outer 3 layers from the array -- later we can add these back with an exterpolator perhaps?

    return acceleration

def FiniteDifferenceMethod(v, dx=1e-5, stencil=5, output="abs"):
    # Compute the shape of the array
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
        print("Incorrect output method for the Finite Difference Method")



# --------------------------------------------------------------------

def DetermineSeedScore(seed, target_contour, target_polynomial, n_steps=100):
    seed_score = 0

    contour_x = target_contour[:, 0]
    contour_y = target_contour[:, 1]

    contour_cutoff_min = contour_x[np.where(contour_y > laminar_height*1.5)[0][0]]
    contour_cutoff_max = contour_x[np.argmax(contour_y)]

    if seed[0] > X.min() and seed[0] < contour_cutoff_min:

        xspace = np.arange(contour_cutoff_min, contour_cutoff_max, n_steps)
        streamline_positions = GetCleanStreamlinePositions(seed, xspace)
        target_positions = target_polynomial(streamline_positions[:, 0])

        seed_score = np.sqrt(np.sum((streamline_positions[:, 1] - target_positions)**2)/len(streamline_positions))
    else:
        seed_score = 1000

    return seed_score

def GenerationalFitting(n_generations=3, print_progress=False, target_method="mean"):
    laminar_height = EstimateLaminarHeight()
    # TODO: AUTOMATE THIS
    xc_min = 0.45
    xc_max = 0.50

    # Obtain contour points and target polynomial
    if print_progress: print("-- Obtaining Target Points --")

    if target_method == "mean":
        target_points, target_polynomial = GetTargetContour()
    elif target_method == "change":
        target_points, target_polynomial = GetTargetChange()
    else:
        print("Incorrect Target Method inserted, please try again")

    if print_progress: print(target_polynomial)
    if print_progress: print(" -- Target obtained --")

    # Generate steps array to work with throughout the generations
    gen_steps = np.logspace(-1, -n_generations, num=n_generations)
    if print_progress: print(gen_steps)

    # For each generation:
    for generation_id in range(n_generations-1):
        if print_progress: 
            print(" ----- Generation: ", generation_id, " -----")
            print("Obtaining Seeds")
        
        # Setup seed array from boundaries
        s = np.arange(xc_min, xc_max, gen_steps[generation_id+1])
        seeds = []
        for seed_xc in s:
            seeds.append([seed_xc, laminar_height])
        seeds = np.array(seeds)

        if print_progress: print("Seeds obtained, finding scores")

        # Generate score for each seed, and compare to best
        best_score = np.inf
        for seed in seeds:
            seed_score = DetermineSeedScore(seed, target_contour=target_points, target_polynomial=target_polynomial)
            if print_progress:  print("Seed: ", seed, " | RMS: ", seed_score , " | Similarity: ", 1/seed_score)
            if seed_score < best_score:
                best_score = seed_score
                gen_seed = seed
        
        if print_progress: 
            print("Generation completed | Best Seed: ", gen_seed, " with score: ", best_score)
            print("Obtaining new Boundaries")

        # Get new boundary value for the next generation
        xc_min = gen_seed[0] - gen_steps[generation_id] * gen_seed[0]/5
        xc_max = gen_seed[0] + gen_steps[generation_id] * gen_seed[0]/5
        if print_progress:  print("Boundaries: ", xc_min, " | ", xc_max)

    print("All Generations completed, Final seed obtained: ", gen_seed, " with Similarity: ", 1/best_score)
    return gen_seed


# --------------------------------------------------------------------------

def PlotTargetHeatmap(seed, target_contour=None, target_polynomial=None):
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

    if target_contour == None or target_polynomial == None:
        target_contour, target_polynomial = GetTargetContour()
    
    target_xspace = np.linspace(target_contour[0,0], target_contour[-1, 0], xcount*heatmap_resolution)
    ax_heatmap.scatter(target_contour[:,0], target_contour[:, 1], color="black", s=0.7)
    ax_heatmap.plot(target_xspace, target_polynomial(target_xspace), "k--", linewidth=0.7)

    plt.show()

def PlotAccelerationHeatmap():
    acceleration = GetAccelerationMap()

    # Plotting the Heatmap
    fig_gradient, ax_gradient = plt.subplots()
    heatmap = ax_gradient.imshow(acceleration[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
    plt.colorbar(heatmap, label="Absolute acceleration [1/U$_{inf}$]", ax=ax_gradient)
    ax_gradient.set_xlabel("x/c [-]")
    ax_gradient.set_ylabel("y/c [-]")

    plt.show()