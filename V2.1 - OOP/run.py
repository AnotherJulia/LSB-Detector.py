from streamline import Streamline
from dataprocessor import DataProcessor
from fitter import Fitter

piv_path = "../PIV.dat"
print("----------------------------------------")

# Initialize the dataprocessor
dataprocessor = DataProcessor(data_path=piv_path, data_resolution=1)

# Plot the Velocity or Acceleration Heatmap
# dataprocessor.plot("acceleration heatmap"

# Initialize a streamline with custom seed
# streamline = Streamline(dataprocessor, seed = [0.47, 0.001])

# Plot the above initialized streamline in a basic streamline plot
# streamline.plot()

# Get the target points and target polynomial using the acceleration map
target_points, target_polynomial = dataprocessor.getTargetContourAcceleration(plot=False)

# Initialize the Fitter object
fitter = Fitter(dataprocessor, target_points, target_polynomial, n_generations=2, initial_seed_xc=0.47, n_steps=10)
fitter.runFittingSequence()
print(fitter.best_seed)