import numpy as np

from dataprocessor import *
from streamline import *


class Fitter():
    def __init__(self, dataprocessor:DataProcessor, target_points, target_polynomial, n_generations = 2, initial_seed_xc=0.47, n_steps=100):
        self.dataprocessor = dataprocessor

        self.n_generations = n_generations
        self.n_steps = n_steps

        self.target_points = target_points
        self.target_polynomial = target_polynomial

        self.current_laminar_height = self.dataprocessor.estimateLaminarHeight(seperation_point_estimate_xc=initial_seed_xc, return_=True)
        self.initial_seed = np.array([initial_seed_xc,self.current_laminar_height])

        self.best_seed = np.array([])

        
    def runFittingSequence(self):
        print("=== Running Fitting Sequence ===")
        xc_min, xc_max = 0.45, 0.50

        gen_steps = np.logspace(-1, -self.n_generations, num=self.n_generations)
        
        for generation_id in range(self.n_generations-1):
            print("--- Generation: 0 ---")
            
            print("Obtaining seeds")
            s = np.arange(xc_min, xc_max, gen_steps[generation_id+1])
            seeds = []
            for xc in s:
                seeds.append([xc, self.current_laminar_height])
            seeds = np.array(seeds)
            print("Seeds obtained, initializing streamlines and finding scores")

            best_score = np.inf
            for seed in seeds:
                seed_score = self.determineSeedScore(seed)
                print("Seed: ", seed, " | RMS: ", seed_score, " | Similarity: ", 1/seed_score)
                if seed_score < best_score:
                    best_score = seed_score
                    gen_seed = seed
            
            print("Generation Completed | Best seed: ", gen_seed, " with similarity: ", 1/best_score)
            
            print("Recalculating the laminar height")
            self.current_laminar_height = self.dataprocessor.estimateLaminarHeight(seperation_point_estimate_xc=gen_seed[0], return_=True)
            print("> New estimated laminar height: ", self.current_laminar_height)
            
            print("Obtaining new boundaries")

            xc_min = gen_seed[0] - gen_steps[generation_id] * gen_seed[0]/5
            xc_max = gen_seed[0] + gen_steps[generation_id] * gen_seed[0]/5
            print("New boundaries: ", xc_min, " | ", xc_max)
        
        print("All generations completed, final seed obtained: ", gen_seed, " with similarity: ", 1/best_score)
        self.best_seed = gen_seed

    def determineSeedScore(self, seed):
        seed_score = 0

        contour_cutoff_min = self.target_points[:,0][np.where(self.target_points[:,1] > self.current_laminar_height*1.5)[0][0]]
        contour_cutoff_max = self.target_points[:,1][np.argmax(self.target_points[:,1])]

        if seed[0] > self.dataprocessor.X.min() and seed[0] < contour_cutoff_min:
            xspace = np.arange(contour_cutoff_max, contour_cutoff_max, self.n_steps)

            # Initialize streamline object
            streamline = Streamline(self.dataprocessor, seed=seed)
            positions = streamline.positions

            '''
            # ! NOTE: Currently working on this
            '''



            target_positions = self.target_polynomial(positions[0,:])

            seed_score = np.sqrt(np.sum((positions[1,:] - target_positions)**2)/len(positions))
        else: seed_score = 1000

        return seed_score