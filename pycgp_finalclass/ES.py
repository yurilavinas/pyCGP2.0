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
import seaborn as sns

class ES: #Evolution strategy
    def __init__(self, evaluator, lam,parent_factory,mutation,config): 
        self.evaluator = evaluator
        self.lam = lam #offspring population size
        self.parent_factory = parent_factory
        self.mutation = mutation
        self.config = config
    
    #evolving process: evolve n time and stopping at a certain point without improvement
    def evolve(self, n_generations, early_stopping,early_switch,project_name= "ES_run", verbose=False): #Put true in verbose to see prints
        parent = self.parent_factory()
        current_genome = parent.copy() #deepcopy to avoid mutating best_genome
        best_genome = current_genome.copy() #deepcopy to avoid mutating best_genome
        best_train_fitness = self.evaluator.evaluate(parent)[0] #start from the lowest value possible
        best_test_fitness = self.evaluator.evaluate(parent)[1]
        print(f"Starting fitness {best_train_fitness:.4f}")
        num_improvement = 0
        num_switch = 0
        
        # List to track best fitness per generation
        fitness_history = []
        evaluation_count = 0

        cumulative_usage = [0] * self.config.num_inputs  # cumulative counter
        feature_usage_over_time = []  # List of dicts: one per generation
    

        pbar = trange(n_generations, desc="Evolving", unit="gen", disable=not verbose)

        for generation in pbar:
            offspring = []
            for i in range(self.lam):
                child = current_genome.copy()
                self.mutation.mutate(child)
                offspring.append(child)
                

            # Evaluate population
            scored_population = []
            generation_fitnesses = []
            for genome in offspring:
                #dont need to evalua
                train_fitness, test_fitness = self.evaluator.evaluate(genome)
                scored_population.append((genome, train_fitness,test_fitness))

                #for fitness plotting
                evaluation_count += 1
                generation_fitnesses.append(train_fitness)
                fitness_history.append((evaluation_count, best_train_fitness,best_test_fitness))

            # Sort the population based on fitness
            scored_population.sort(key=lambda x: x[1], reverse=True)


            # Update best genome if fitness improves
            if scored_population[0][1] > best_train_fitness:
                best_train_fitness = scored_population[0][1]
                best_test_fitness = scored_population[0][2]
                best_genome = scored_population[0][0].copy() #deepcopy to avoid mutating best_genome
                current_genome = best_genome.copy()  # Update the used genome to the best found

                used_input_counts = self.get_used_input_features(best_genome)
                cumulative_usage = [cumul + used for cumul, used in zip(cumulative_usage, used_input_counts)]
                feature_usage_over_time.append(cumulative_usage.copy())  # snapshot of current state

                num_improvement = 0
                num_switch = 0
                if verbose:
                    # Print the top individual function string
                    pbar.set_description(f"Gen {generation} | Best: {best_train_fitness:.4f}")
            else:
                num_improvement += 1
                num_switch += 1
            if num_switch >= early_switch:
                current_genome = self.parent_factory()
                num_switch = 0
            if num_improvement >= early_stopping:
                pbar.set_description(f"Early Stop at Gen {generation}")
                break

        # Final output
        print(f"\nBest Train fitness achieved: {best_train_fitness:.4f}")
        print(f"\nTest fitness: {best_test_fitness:.4f}")
        print(best_genome.to_function_string())
        self.plot_fitness_convergence(fitness_history)
        best_genome.visualize_active_graph()
        self.log_run_result(best_genome, best_test_fitness, generation,project_name, log_dir="Results")
        self.plot_feature_usage(feature_usage_over_time)
        return best_genome
       
    def plot_fitness_convergence(self, fitness_history):
        evaluations, best_train_fitnesses,best_test_fitnesses = zip(*fitness_history)

        plt.figure(figsize=(10, 6))
        plt.plot(evaluations, best_train_fitnesses, label='Best-so-Far Train Fitness', color='blue', linewidth=1.5)
        plt.plot(evaluations, best_test_fitnesses, label='Test Fitness (of Best Train)', color='green', linestyle='--', linewidth=1.5)
        plt.title('Fitness Convergence Over Evaluations')
        plt.xlabel('Evaluation Count')
        plt.ylabel('Fitness')
        plt.ylim(-2, 2)
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


    def get_used_input_features(self,genome):
        num_inputs = self.config.num_inputs
        used_inputs = [0] * num_inputs
        for node in genome.get_active_nodes():
            for input_idx in node.inputs:
                if input_idx < num_inputs:  # input node index
                    used_inputs[input_idx] += 1
        return used_inputs


    def plot_feature_usage(self, feature_usage_over_time, top_n=8):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        usage_array = np.array(feature_usage_over_time).T  # shape: [n_features, n_generations]
        generations = list(range(len(feature_usage_over_time)))

        # Step 1: Compute total usage per feature
        total_usage = usage_array.sum(axis=1)

        # Step 2: Get indices of top N features
        top_indices = np.argsort(total_usage)[-top_n:][::-1]  # descending order

        # Step 3: Line plot for top features
        plt.figure(figsize=(12, 6))
        for i in top_indices:
            plt.plot(generations, usage_array[i], label=f'Feature {i}')
        plt.xlabel("Generation")
        plt.ylabel("Usage Count")
        plt.title(f"Top {top_n} Feature Usage Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Step 4: Correlation heatmap for top features
        df_usage = pd.DataFrame(usage_array[top_indices].T, columns=[f"Feature_{i}" for i in top_indices])
        df_diff = df_usage.diff().dropna()
        corr = df_diff.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title(f"Correlation of Top {top_n} Feature Usage Changes")
        plt.tight_layout()
        plt.show()

