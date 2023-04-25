from streamline import Streamline
from dataprocessor import DataProcessor
from fitter import Fitter

piv_path = "../PIV.dat"
print("----------------------------------------")

# Initialize the dataprocessor
dataprocessor = DataProcessor(data_path=piv_path, data_resolution=10, acceleration_resolution=2)

# Plot the Velocity or Acceleration Heatmap
# dataprocessor.plot("acceleration heatmap"

# Initialize a streamline with custom seed
# streamline = Streamline(dataprocessor, seed = [0.47, dataprocessor.laminar_height])

# Plot the above initialized streamline in a basic streamline plot
# streamline.plot()

# Get the target points and target polynomial using the acceleration map
target_points, target_polynomial = dataprocessor.getTargetContourAcceleration(plot=False)

# Plot the target points in an acceleration heatmap
# dataprocessor.plotTargetPolynomial(target_polynomial)

print("----------------------------------------")

# Initialize the Fitter object
fitter = Fitter(dataprocessor, target_points, target_polynomial, n_generations=3, initial_seed_xc=0.47, n_steps=10)
fitter.runFittingSequence()

# Initialize the best streamline from the fitter seed and plot it 
streamline = Streamline(dataprocessor, seed=fitter.best_seed)
streamline.plot()

# Retrieve and print the streamline characteristics
seperation, reattachment, transition = streamline.retrieveCharacteristics(decimals=4)
print(f"Seperation: {seperation} | Reattachment: {reattachment} | Transition: {transition}")

# Plot the boundary layer thickness graph
streamline.plotBoundaryLayerThickness()

