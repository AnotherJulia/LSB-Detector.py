from streamline import Streamline
from dataprocessor import DataProcessor
from fitter import Fitter

piv_path = "../PIV.dat"
print("----------------------------------------")

# Initialize the dataprocessor
dataprocessor = DataProcessor(data_path=piv_path, data_resolution=2, acceleration_resolution=2)

# Plot the Velocity and Acceleration Heatmap
# dataprocessor.plot("velocity heatmap --interpolate", resolution=10, save_image=True)
# dataprocessor.plot("acceleration heatmap --interpolate", resolution=10, save_image=True)

# Initialize a streamline with custom seed
# streamline = Streamline(dataprocessor, seed = [0.47, dataprocessor.laminar_height])

# Plot the above initialized streamline in a basic streamline plot
# streamline.plot()

# Get the target points and target polynomial using the acceleration map
target_points, target_polynomial = dataprocessor.getTargetContourAcceleration(plot=False)

# Plot the target points in an acceleration heatmap
# dataprocessor.plotTargetPolynomial(target_polynomial, save_image=True)

print("----------------------------------------")

# Initialize the Fitter object
fitter = Fitter(dataprocessor, target_points, target_polynomial, n_generations=3, initial_seed_xc=0.47, n_steps=10)
fitter.runFittingSequence()

# Initialize the best streamline from the fitter seed and plot it 
streamline = Streamline(dataprocessor, seed=fitter.best_seed)

# Plot the streamlines on top of the velocity and velocity gradient maps
# streamline.plot(plot_type="streamline velocity", save=True)
# streamline.plot(plot_type="streamline velocity gradient", save=True)

# Retrieve and print the streamline characteristics
seperation, reattachment, transition = streamline.retrieveCharacteristics(decimals=5)
print(f"Seperation: {seperation} | Transition: {transition} | Reattachment: {reattachment} | Measurement Uncertainty: ± 0.0009")

# Plot the boundary layer thickness graph
# streamline.plotBoundaryLayerThickness()

