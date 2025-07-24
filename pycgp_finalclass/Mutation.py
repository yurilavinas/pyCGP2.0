from abc import ABC, abstractmethod
import random
import numpy as np

class Mutation(ABC): #Abstract class for mutation
    @abstractmethod
    def mutate(self, genome):
        pass

class Golden_mutation(Mutation): 

    def __init__(self,config, function_mutation_rate,input_mutation_rate,const_mutation_rate,output_node_mutation_rate,sa=True):
        self.config = config
        self.function_mutation_rate = function_mutation_rate
        self.input_mutation_rate = input_mutation_rate
        self.const_mutation_rate = const_mutation_rate
        self.output_node_mutation_rate = output_node_mutation_rate

        self.suc = 0
        if sa:
            self.sa=True
            self.up = up = 1/np.sqrt(config.num_inputs+1)
        self.suc_in = 0
        self.suc_out = 0
        self.suc_con = 0
        self.suc_f = 0
    
    def mutate(self, genome):

        has_functionnaly_mutated = False

        active_node_indices = {node.index for node in genome.get_active_nodes()}

        while not has_functionnaly_mutated:
            # Apply one mutation
            r = random.random()

            if self.sa:
                self.suc_in = 0
                self.suc_out = 0
                self.suc_con = 0
                self.suc_f = 0
            # Chance of mutating output
            node = random.choice(genome.nodes)
            node_index = node.index

            if r < self.output_node_mutation_rate:
                self.mutate_outputs(genome)
                self.suc_out = 1
            elif r < self.function_mutation_rate:
                node.mutate_function(self.config.function_set, node_index)
                self.suc_f = 1
            elif r < self.input_mutation_rate:
                node.mutate_inputs(self.config.num_inputs, node_index)
                self.suc_in = 1
            elif r < self.const_mutation_rate:
                node.mutate_constants(self.config.const_min, self.config.const_max)
                self.suc_con = 1
            
            
            # If the mutated node is active, we stop
            has_functionnaly_mutated = len(active_node_indices) != len({node.index for node in genome.get_active_nodes()})

        if self.sa:
            self.SA()

    #Mutate the outputs of the genome(used in mutate)
    def mutate_outputs(self, genome):
        # Pick a random output index to mutate
        output_index = random.randint(0, len(genome.outputs) - 1)
        current_output = genome.outputs[output_index]
        
        possible_outputs = list(set(range(self.config.num_inputs, self.config.num_inputs + self.config.num_nodes)) - {current_output})
        if possible_outputs:
            new_output = random.choice(possible_outputs)
            genome.outputs[output_index] = new_output

    def SA(self):
        # from https://cw.fel.cvut.cz/old/_media/courses/a0m33eoa/prednasky/eoa03_realeas_handouts.pdf    
        if self.suc_f:
            self.function_mutation_rate *= np.exp(self.suc*self.suc_f - 1/5)**self.up
        if self.suc_in:
            self.input_mutation_rate *=  np.exp(self.suc*self.suc_in - 1/5)**self.up
        if self.suc_con:
            self.const_mutation_rate *= np.exp(self.suc*self.suc_con - 1/5)**self.up
        if self.suc_out:
            self.output_node_mutation_rate *= np.exp(self.suc*self.suc_out - 1/5)**self.up

        self.function_mutation_rate = np.clip(0.001,self.function_mutation_rate,0.5)
        self.input_mutation_rate = np.clip(0.001,self.input_mutation_rate,0.5)
        self.const_mutation_rate = np.clip(0.001,self.const_mutation_rate,0.5)
        self.output_node_mutation_rate = np.clip(0.001,self.output_node_mutation_rate,0.5)
        
