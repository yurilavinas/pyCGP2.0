from abc import ABC, abstractmethod
import random
import copy
from pycgp_finalclass.Node import Node
from pycgp_finalclass.Function import Func
from pycgp_finalclass.Config import CGPConfig
from pycgp_finalclass.Config import Config
import numpy as np

class Genome(ABC): #Abstract class for our genome
    @abstractmethod
    def copy(self):
        pass

    @classmethod
    @abstractmethod
    def create_genome(cls, config):
        pass

class CGPGenome: #This class contains every function that apply directly to the genome
    def __init__(self, config, nodes=None,):
        self.config = config
        self.nodes = nodes if nodes is not None else self._init_nodes() #you can pass in the parameters a specific genome
        #Choose n random outputs from the list of internal nodes
        self.outputs = [
            random.choice(
                range(config.num_inputs, config.num_inputs + config.num_nodes)
            )
            for _ in range(config.num_outputs)
        ]

    def copy(self):
        return copy.deepcopy(self)      

    @classmethod
    def create_genome(cls, config):
        return cls(config)

    #Initialise a list of nodes of size num_nodes
    def _init_nodes(self): 
        nodes = []
        for i in range(self.config.num_nodes):
            cgp_func = random.choice(self.config.function_set) #Assign a random function from the function set

            const_params = [ #Random uniform values for the constants
                random.uniform(self.config.const_min, self.config.const_max)
                for _ in range(cgp_func.const_params)
            ]

            max_index = self.config.num_inputs + i   # Max valid input index
            inputs = []

            for _ in range(cgp_func.arity):
                if max_index <= self.config.num_inputs:
                    # No internal nodes available yet
                    inputs.append(random.randint(0, self.config.num_inputs - 1))
                else:
                    if random.random() < self.config.node_input_chance:
                        inputs.append(random.randint(0, self.config.num_inputs - 1))  # Input node
                    else:
                        inputs.append(random.randint(self.config.num_inputs, max_index-1))  # Previous node only

            nodes.append(Node(cgp_func, inputs, const_params, max_index)) #fill the list of nodes of the genome
        return nodes


    def get_value(self, input_values):
        values = {i: val for i, val in enumerate(input_values)}  # dictionary of input values

        active_nodes = self.get_active_nodes() #Only compute the active nodes

        for node in active_nodes:
            values[node.index] = node.execute(values) # execute function for each active node to get the end values

        return [values[i] for i in self.outputs] # get values for the outputs nodes


    #used to display the genome in an understandable way
    def to_function_string(self):
        func_str = ""

        # Generate the function string for each node
        for node in self.nodes:
            inputs = [f"x{idx}" if idx < self.config.num_inputs else f"n{idx}" for idx in node.inputs]
            func = node.Func.name
            func_str += f"n{node.index} = {func}({', '.join(inputs)})\n"

        # Show output references
        func_str += "Outputs: " + ", ".join([f"n{idx}" for idx in self.outputs]) + "\n\n"

        # Helper to recursively unroll a node
        def unroll_node(idx):
            if idx < self.config.num_inputs:
                return f"x{idx}"
            else:
                node = self.nodes[idx - self.config.num_inputs]
                func = node.Func.name
                args = [unroll_node(i) for i in node.inputs]
                return f"{func}({', '.join(args)})"

        # Unroll all outputs
        for i, out_idx in enumerate(self.outputs):
            output_expr = unroll_node(out_idx)
            func_str += f"Unrolled output expression {i}:\n{output_expr}\n"

        return func_str



    #Get a list of nodes connected to the outputs(including the outputs)
    def get_active_nodes(self):
        # Create a node dictionary for quick access
        index_to_node = {node.index: node for node in self.nodes}

        # start with the output nodes
        active_indices = set(self.outputs)
        changed = True

        while changed:
            changed = False
            for node_idx in list(active_indices):
                node = index_to_node.get(node_idx)
                if node is None:
                    # if the node is not found, skip it
                    continue
                for input_idx in node.inputs:
                    # if the node is not active, add it to the active set
                    if input_idx >= self.config.num_inputs and input_idx not in active_indices:
                        if input_idx in index_to_node:
                            active_indices.add(input_idx)
                            changed = True
                        else:
                            # debug
                            print(f"[WARNING] input_idx {input_idx} not found in nodes.")

        # return actives nodes sorted by index
        active_nodes = [index_to_node[i] for i in active_indices if i in index_to_node]
        active_nodes.sort(key=lambda node: node.index)

        return active_nodes


    def visualize_active_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        def compute_node_depths(active_nodes, num_inputs):
            depths = {}
            node_map = {node.index: node for node in active_nodes}

            def get_depth(index):
                if index < num_inputs:
                    return 0
                if index in depths:
                    return depths[index]
                node = node_map[index]
                input_depths = [get_depth(i) for i in node.inputs if i < num_inputs or i in node_map]
                d = 1 + (max(input_depths) if input_depths else 0)
                depths[index] = d
                return d

            for node in active_nodes:
                get_depth(node.index)

            return depths

        active_nodes = self.get_active_nodes()
        active_node_indices = {node.index for node in active_nodes}
        active_output_indices = [idx for idx in self.outputs if idx in active_node_indices or idx < self.config.num_inputs]

        G = nx.DiGraph()
        pos = {}
        labels = {}

        layer_spacing = 3.0
        vertical_spacing = 2.0

        # --- INPUT NODES ---
        active_input_indices = sorted({i for node in active_nodes for i in node.inputs if i < self.config.num_inputs})
        for i, idx in enumerate(active_input_indices):
            node_id = f"x{idx}"
            x = 0
            y = -i * vertical_spacing + (len(active_input_indices) - 1) * vertical_spacing / 2
            pos[node_id] = (x, y)
            labels[node_id] = node_id
            G.add_node(node_id, color='lightblue')

        # --- INTERNAL NODES by DEPTH ---
        depths = compute_node_depths(active_nodes, self.config.num_inputs)
        depth_groups = {}
        for node in active_nodes:
            d = depths[node.index]
            depth_groups.setdefault(d, []).append(node)

        sorted_depths = sorted(depth_groups.keys())
        max_depth = max(sorted_depths)

        for d in sorted_depths:
            nodes_at_depth = depth_groups[d]
            for i, node in enumerate(nodes_at_depth):
                node_id = f"n{node.index}"
                x = (d + 1) * layer_spacing  # internal nodes start at x=1
                y = -i * vertical_spacing + (len(nodes_at_depth) - 1) * vertical_spacing / 2
                pos[node_id] = (x, y)
                labels[node_id] = f"n{node.index}\n{node.Func.name}"
                G.add_node(node_id, color='lightgreen')

        # --- OUTPUT NODES ---
        for i, idx in enumerate(active_output_indices):
            output_id = f"out{i}"
            x = (max_depth + 2) * layer_spacing  # outputs are after last depth
            y = -i * vertical_spacing + (len(active_output_indices) - 1) * vertical_spacing / 2
            pos[output_id] = (x, y)
            labels[output_id] = f"y{i}"
            G.add_node(output_id, color='orange')

            # Link output to its source
            if idx < self.config.num_inputs:
                G.add_edge(f"x{idx}", output_id)
            else:
                G.add_edge(f"n{idx}", output_id)

        # --- EDGES from inputs/internals to internal nodes ---
        for node in active_nodes:
            target_id = f"n{node.index}"
            for input_idx in node.inputs:
                if input_idx < self.config.num_inputs:
                    source_id = f"x{input_idx}"
                elif input_idx in active_node_indices:
                    source_id = f"n{input_idx}"
                else:
                    continue  # skip inactive nodes
                G.add_edge(source_id, target_id)

        # --- DRAW ---
        node_colors = [G.nodes[n].get('color', 'gray') for n in G.nodes]
        nx.draw(G, pos, with_labels=True, labels=labels,
                node_color=node_colors, node_size=500,
                font_size=8, arrows=True, edge_color='gray')

        plt.title("Active Genome Graph (Inputs → Internal → Outputs)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


