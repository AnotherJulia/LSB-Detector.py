from streamline import Streamline
from dataprocessor import DataProcessor

piv_path = "../PIV.dat"

# Initialize the dataprocessor
dataprocessor = DataProcessor(data_path=piv_path, data_resolution=10)

# Initialize a streamline
# streamline = Streamline(dataprocessor, seed = [0.47, 0.001])

# Plot the above initialized streamline in a basic streamline plot
# streamline.plot()
