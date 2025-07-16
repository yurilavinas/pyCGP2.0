from abc import ABC, abstractmethod
import random
class Mutation(ABC): #Abstract class for mutation
    @abstractmethod
    def mutate(self, genome):
        pass

class Golden_mutation(Mutation): 

    def __init__(self,config,input_node_mutation_rate, function_mutation_rate,input_mutation_rate,const_mutation_rate,output_node_mutation_rate):
        self.config = config
        self.input_node_mutation_rate = input_node_mutation_rate
        self.function_mutation_rate = function_mutation_rate
        self.input_mutation_rate = input_mutation_rate
        self.const_mutation_rate = const_mutation_rate
        self.output_node_mutation_rate = output_node_mutation_rate


    def mutate(self, genome):

        def snapshot_structure(genome):
            return [
                (n.index, n.Func.name, tuple(n.inputs), tuple(getattr(n, 'constants', [])))
                for n in genome.get_active_nodes()
            ]
        # Chance of mutating output
        if random.random() < self.output_node_mutation_rate:
            self.mutate_outputs(genome)


        while True:
            node = random.choice(genome.nodes)
            node_index = node.index

            active_before = snapshot_structure(genome)
            # Apply one mutation
            r = random.random()
            if r < self.function_mutation_rate:
                node.mutate_function(self.config.function_set, node_index)
            elif r < self.function_mutation_rate + self.input_mutation_rate:
                node.mutate_inputs(self.config.num_inputs, node_index, self.input_node_mutation_rate)
            else:
                node.mutate_constants(self.config.const_min, self.config.const_max)

            active_after = snapshot_structure(genome)

            if node_index in {n.index for n in genome.get_active_nodes()} and active_before != active_after:
                break


    #Mutate the outputs of the genome(used in mutate)
    def mutate_outputs(self, genome):
        # Pick a random output index to mutate
        output_index = random.randint(0, len(genome.outputs) - 1)
        current_output = genome.outputs[output_index]
        
        possible_outputs = list(set(range(self.config.num_inputs, self.config.num_inputs + self.config.num_nodes)) - {current_output})
        if possible_outputs:
            new_output = random.choice(possible_outputs)
            genome.outputs[output_index] = new_output

