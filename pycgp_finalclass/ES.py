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

class ES: #Evolution strategy
    def __init__(self, evaluator, lam,parent,mutation): 
        self.evaluator = evaluator
        self.lam = lam #offspring population size
        self.parent = parent
        self.mutation = mutation
    
    #evolving process: evolve n time and stopping at a certain point without improvement
    def evolve(self, n_generations, early_stopping,project_name= "ES_run", verbose=False): #Put true in verbose to see prints
        parent = self.parent
        best_genome = self.parent #deepcopy to avoid mutating best_genome
        best_fitness = self.evaluator.evaluate(parent) #start from the lowest value possible
        print(f"Starting fitness {best_fitness:.4f}")
        no_improvement = 0
        
        # List to track best fitness per generation
        fitness_history = []
        mean_std_history = []
        evaluation_count = 0    

        for generation in range(n_generations):
            if verbose:
                print(f"Generation {generation}")
            offspring = []
            for i in range(self.lam):
                parent = best_genome.copy() #deepcopy to avoid mutating best_genome
                child = parent.copy()
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
                no_improvement = 0
                if verbose:
                    # Print the top individual function string
                    print(f"\nBest fitness this generation: {best_fitness:.4f}")
                    print(best_genome.to_function_string())
            else:
                no_improvement += 1

            if no_improvement >= early_stopping:
                print(f"Early stopping at generation {generation} (no improvement for {early_stopping} generations).")
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
