from main import *
from targeting import *
from transition import *

# ------------------------
seed = RunTargeting(n_generations=1, print_progress=True, target_method="mean")
#seperation, reattachment, transition = GetCharacteristics(seed, print_characteristics=False)
#PlotTargetHeatmap(seed)

GetTargetChange(plot=True)

