from streamline import Streamline
from dataprocessor import DataProcessor

piv_path = "../PIV.dat"

# Initialize the dataprocessor
dataprocessor = DataProcessor(data_path=piv_path, data_resolution=2)

# Plot the Velocity or Acceleration Heatmap
# dataprocessor.plot("acceleration heatmap"

# Initialize a streamline with custom seed
# streamline = Streamline(dataprocessor, seed = [0.47, 0.001])

# Plot the above initialized streamline in a basic streamline plot
# streamline.plot()

# Get the target points and target polynomial using the acceleration map
_, target_polynomial = dataprocessor.getTargetContourAcceleration(plot=False)

# Initialize the Fitter object
 