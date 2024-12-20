import torch
from torch import Tensor
from lib.Solver import Solver
from models.mdds import Net
import pickle 
import networkx as nx
import csv 


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait before stopping when validation loss is not improving.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss >= self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


class DNNMDDS(Solver):
    """
    A solver class for finding the Maximum Independent Set (MIS) of a graph using a
    dataless neural network model. The neural network is trained to predict theta values
    which are then used to determine the MIS.

    Parameters:
        G (networkx.Graph): The graph on which the MIS problem will be solved.
        params (dict): Dictionary containing solver parameters:
            - max_steps (int, optional): Maximum number of training steps for the model. Defaults to 100000.
            - selection_criteria (float, optional): Threshold for selecting nodes based on theta values. Defaults to 0.5.
            - learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
            - use_cpu (bool, optional): Flag to use CPU for computations instead of GPU. Defaults to False.
    """

    def __init__(self, filename, params = {}):
        """
        Initializes the DNNMIS solver with the given graph and parameters.

        Args:
            G (networkx.Graph): The graph to solve the MIS problem on.
            params (dict): Parameters for the solver including max_steps, selection_criteria, learning_rate, and use_cpu.
        """
        super().__init__()
        self.selection_criteria = params.get("selection_criteria", 0.5)
        self.learning_rate = params.get("learning_rate", 0.0001)
        self.max_steps = params.get("max_steps", 100000)
        self.use_cpu = params.get("use_cpu", False)
        params['theta_init'] = params.get('theta_init', [])
        self.graph_name = filename 
        G = pickle.load(open(self.graph_name, 'rb'))

        self.graph = G
        self.vertices = len(G.nodes)
        self.edges = len(G.edges)
        print(self.vertices)

        self.model = Net(G, theta_init=params['theta_init'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = lambda predicted, desired: predicted - desired
        
        self.x = torch.ones(self.vertices)
        self.objective = torch.tensor(-(self.vertices) / 2)
        self.solution = {}

        self.early_stopping = EarlyStopping()
        self.columns = ["nodes", "edges", "iterations", "graph_type", "graph_filename", "mdds", "time"]
        self.csv_filename = 'dNNMDDS_benchmark.csv'

    def solve(self):
        """
        Trains the neural network model to find the Maximum Independent Set (MIS) of the graph.

        The method performs the following steps:
        1. Trains the model for a specified number of steps.
        2. Evaluates the model to get theta values.
        3. Applies a selection criterion to determine which nodes are in the MIS.
        4. Constructs the MIS by iteratively removing nodes with the highest degree from the subgraph.
        5. Records the solution details including the graph mask, size of the MIS, and number of training steps.

        Outputs:
            - self.solution (dict): Contains the results of the MIS computation:
                - graph_mask (list of int): List where 1s denote nodes in the MIS.
                - graph_probabilities (list of float): Theta weight results for each node in the graph.
                - size (int): Size of the MIS.
                - number_of_steps (int): Number of training steps performed.
                - steps_to_best_MIS (int): Number of steps to reach the best MIS (currently set to 0).
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu")
        print("using device: ", device)

        self.model = self.model.to(device)
        self.x = self.x.to(device)
        self.objective = self.objective.to(device)

        self._start_timer()
        iterations = 0

        for i in range(self.max_steps):
            iterations = i
            self.optimizer.zero_grad()

            predicted: Tensor = self.model(self.x)

            output = self.loss_fn(predicted, self.objective)

            output.backward()
            self.optimizer.step()

            if i % 500 == 0:
                self.early_stopping(predicted)
                if self.early_stopping.early_stop:
                    print('Stopping Early due to loss plateau')
                    break 
                print(
                    f"Training step: {i}, Output: {predicted.item():.4f}, Desired Output: {self.objective.item():.4f}"
                )

        self._stop_timer()

        self.solution["graph_probabilities"] = self.model.theta_layer.weight.detach().tolist()

        graph_mask = [0 if x < self.selection_criteria else 1 for x in self.solution["graph_probabilities"]]
        indices = [i for i, x in enumerate(graph_mask) if x == 1]

        subgraph = self.graph.subgraph(indices)
        subgraph = nx.Graph(subgraph)
        while len(subgraph) > 0:
            degrees = dict(subgraph.degree())
            max_degree_nodes = [
                node
                for node, degree in degrees.items()
                if degree == max(degrees.values())
            ]

            if (
                len(max_degree_nodes) == 0
                or subgraph.degree(max_degree_nodes[0]) == 0
            ):
                break  # No more nodes to remove or all remaining nodes have degree 0

            subgraph.remove_node(max_degree_nodes[0])
        IS_size = len(subgraph)
        MIS_size = IS_size
        MIS_mask = graph_mask
        print(f"Found MDDS of size: {MIS_size}")
        print(f"theta vector: {MIS_mask}")

        self.solution["graph_mask"] = MIS_mask
        self.solution["size"] = MIS_size
        self.solution["number_of_steps"] = i
        self.solution["steps_to_best_MIS"] = 0
        self.solution['convergence_time'] = self.solution_time

        data = [{
            "nodes": self.vertices,
            "edges": self.edges,
            "iterations": iterations,
            "graph_type": 'gnm',
            "graph_filename": self.graph_name,
            "mdds": indices, 
            "time": self.solution_time 
        }]

        with open(self.csv_filename, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writerows(data)  
            print(data)
        print('Done')