import random
import numpy as np
from pycgp_finalclass.Genome import Genome
from pycgp_finalclass.Genome import CGPGenome
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import copy
import os
import pickle
from tqdm import trange  # or tqdm if you want more control

class ES: #Evolution strategy
    def __init__(self, evaluator, lam,parent_factory,mutation): 
        self.evaluator = evaluator
        self.lam = lam #offspring population size
        self.parent_factory = parent_factory
        self.mutation = mutation
    
    #evolving process: evolve n time and stopping at a certain point without improvement
    def evolve(self, n_generations, early_stopping,early_switch,project_name= "ES_run", verbose=False): #Put true in verbose to see prints
        parent = self.parent_factory()
        oui = parent.to_function_string() #to see the function string of the parent genome
        used_genome = parent.copy() #deepcopy to avoid mutating best_genome
        best_genome = used_genome.copy() #deepcopy to avoid mutating best_genome
        best_fitness = self.evaluator.evaluate(parent) #start from the lowest value possible
        print(f"Starting fitness {best_fitness:.4f}")
        no_improvement = 0
        no_switch = 0
        
        # List to track best fitness per generation
        fitness_history = []
        mean_std_history = []
        evaluation_count = 0    

        pbar = trange(n_generations, desc="Evolving", unit="gen", disable=not verbose)

        for generation in pbar:
            offspring = []
            for i in range(self.lam):
                child = used_genome.copy()
                self.mutation.mutate(child)
                offspring.append(child)
                

            # Evaluate population
            scored_population = []
            generation_fitnesses = []
            for genome in offspring:
                #dont need to evalua
                fitness = self.evaluator.evaluate(genome) # put the number of generations
                scored_population.append((genome, fitness))
                #for fitness plotting
                evaluation_count += 1
                generation_fitnesses.append(fitness)
                fitness_history.append((evaluation_count, best_fitness))

            # Sort the population based on fitness
            scored_population.sort(key=lambda x: x[1], reverse=True)

            mean_std_history.append((
                evaluation_count, 
                np.mean(generation_fitnesses), 
                np.std(generation_fitnesses)
            ))

            # Update best genome if fitness improves
            if scored_population[0][1] > best_fitness:
                best_fitness = scored_population[0][1]
                best_genome = scored_population[0][0].copy() #deepcopy to avoid mutating best_genome
                used_genome = best_genome.copy()  # Update the used genome to the best found
                no_improvement = 0
                no_switch = 0
                if verbose:
                    # Print the top individual function string
                    pbar.set_description(f"Gen {generation} | Best: {best_fitness:.4f}")
            else:
                no_improvement += 1
                no_switch += 1
            if no_switch >= early_switch:
                used_genome = self.parent_factory()
                no_switch = 0
                if used_genome.to_function_string() != oui:
                    print("different genome")
            if no_improvement >= early_stopping:
                pbar.set_description(f"Early Stop at Gen {generation}")
                break

        # Final output
        print(f"\nBest fitness achieved: {best_fitness:.4f}")
        print(best_genome.to_function_string())
        self.plot_fitness_convergence(fitness_history,mean_std_history)
        best_genome.visualize_active_graph()
        self.log_run_result(best_genome, best_fitness, generation,project_name, log_dir="Results")
        return best_genome
       
    def plot_fitness_convergence(self, fitness_history,mean_std_history):
        evaluations, best_fitnesses = zip(*fitness_history)
        generations, means, stds = zip(*mean_std_history)

        plt.figure(figsize=(10, 6))
        plt.plot(evaluations, best_fitnesses, label='Best-so-Far Fitness', color='blue', linewidth=1.5)
        plt.plot(generations, means, label='Mean Fitness per evaluation', color='orange', linestyle='--')
        plt.fill_between(generations,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color='orange', alpha=0.2, label='±1 Std Dev')

        plt.title('Fitness Convergence Over Evaluations')
        plt.xlabel('Evaluation Count')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()






    def log_run_result(self,genome, fitness, n_generations,project_name, log_dir="Results"):
        os.makedirs(log_dir, exist_ok=True)

        # Save genome object as a pickle file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pickle_path = os.path.join(log_dir, f"genome_{timestamp}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(genome, f)

        # Prepare log entry
        log_csv = os.path.join(log_dir, f"cgp_results_log_{str(project_name)}.csv")
        entry = {
            "timestamp": timestamp,
            "fitness": fitness,
            "pickle_path": pickle_path,
            "function_string": genome.to_function_string().replace("\n", " | "),  # Make it single-line
            "number of generations": n_generations  
        }

        # Append to CSV (create file if it doesn't exist)
        df_entry = pd.DataFrame([entry])
        if os.path.exists(log_csv):
            df_entry.to_csv(log_csv, mode='a', header=False, index=False)
        else:
            df_entry.to_csv(log_csv, index=False)
