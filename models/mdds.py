import torch 
from torch import nn 
import networkx as nx 

class Hadamard(nn.Module):
    def __init__(self, in_feature, low_bound:int=0, up_bound:int=1):
        super().__init__()
        self.low_bound = low_bound 
        self.up_bound = up_bound
        self.weight = nn.Parameter(torch.Tensor(in_feature))
    
    def forward(self, x):
        self.weight.data = self.weight.data.clamp(self.low_bound, self.up_bound)
        return x * self.weight
    
class Negation(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaler = nn.Parameter(torch.Tensor([-1.0]))
    
    def forward(self, x):
        return x * self.scaler

class MinPool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        k = x.shape[-1]
        if k%3 != 0:
            raise ValueError("Shape not divisible by 3")
        n = k // 3 
        x = torch.cat((x[:n], torch.minimum(x[-n:], x[n:-n])))
        return x 

class Net(nn.Module):
    def __init__(self, G, theta_init=None):
        super().__init__()
        self.graph = G
        self.n = len(self.graph.nodes)
        self.temperature = 0.5
        self.theta_init = theta_init

        self.theta_layer = Hadamard(self.n)
        self.theta_layer.weight = self.theta_weight()
        
        self.layer2 = nn.Linear(in_features=self.n, out_features=3*self.n)
        self.layer2.weight = self.layer2_weight()
        self.layer2.bias = self.layer2_bias()

        self.layer3 = nn.Linear(in_features=2*self.n, out_features=1, bias=False)
        self.layer3.weight = self.layer3_weight()

        self.negative = Negation()
        self.pooling = MinPool()
        
        self.layer2.weight.requires_grad_(False) 
        self.layer2.bias.requires_grad_(False)
        self.layer3.weight.requires_grad_(False)
        self.negative.scaler.requires_grad_(False) 

        self.activation = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
        #     self.layer3.weight.data[self.n + self.m :] = (
        #         -1.0 * self.temperature
        #     )
            self.layer2.bias.data[0 : self.n] = -1.0 * self.temperature

        x = self.theta_layer(x)
        x = self.layer2(x)
        x = self.negative(x)
        x = self.pooling(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x 
    
    def CMSMC_init(self):
        F = {}
        for node in self.graph.nodes:
            f = [ngh for ngh in self.close_ngh(node) if ngh != node for _ in range(2)]
            f.extend(self.scnd_ngh(node))
            F[node] = f
        x = list(self.graph.nodes)
        print(F)
        C = self.key_cover(F, x)
        print(f'heuristic to C: {C}')
        self.theta_init = C 

    def key_cover(self, F, x):
        # Create a copy of the original list to track coverage
        target_elements = x * 2
        
        # Count occurrences of elements in the target list
        target_count = {}
        for elem in target_elements:
            target_count[elem] = target_count.get(elem, 0) + 1
        
        # Selected keys for the cover
        selected_keys = set()
        
        # Remaining elements to be covered
        remaining_count = target_count.copy()
        
        # Greedy approach to select keys
        while remaining_count:
            # Find the key that covers the most remaining elements
            best_key = max(F.keys(), key=lambda k: sum(
                min(F[k].count(elem), remaining_count.get(elem, 0)) 
                for elem in set(F[k])
            ))
            
            # Add the best key to selected keys
            selected_keys.add(best_key)
            
            # Update remaining count by removing covered elements
            for elem in F[best_key]:
                if elem in remaining_count:
                    remaining_count[elem] -= min(F[best_key].count(elem), remaining_count[elem])
                    if remaining_count[elem] <= 0:
                        del remaining_count[elem]
        
        return selected_keys

    def theta_weight(self):
        """ theta parameter vector """
        # self.CMSMC_init()
        g = torch.manual_seed(123)
        theta_weight = torch.rand((self.n), generator=g)

        if self.theta_init:
            theta_weight = theta_weight * 0.5
            theta_weight[self.theta_init] = 0.6 + theta_weight[self.theta_init] * 0.2

        return nn.Parameter(theta_weight)

    def layer2_weight(self):
        """ W matrix derived from G """
        W = torch.zeros([self.n, 3*self.n])
        W[:, :self.n] = torch.eye(self.n)
        for i in range(self.n):
            W[:, self.n+i] = self.one_hot(self.close_ngh(i))
        for i in range(self.n):
            W[:, 2*self.n+i] = self.one_hot(self.scnd_ngh(i))

        return nn.Parameter(W.T.to_dense())

    def layer2_bias(self):
        """ b vector derived from G """
        bias = torch.zeros(3*self.n)
        bias[:self.n] = -0.5
        bias[self.n:2*self.n] = -1.0
        bias[-self.n:] = -2.0

        return nn.Parameter(bias)

    def layer3_weight(self):
        """ w vector derived from G """
        w = torch.zeros(2*self.n) + -1.0
        w[self.n:] = self.n
        
        return nn.Parameter(w.to_dense())
    
    def trainable_params(self):
        """ Counting trainable parameters """
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'trainable params: {params}')
        return params
    
    def close_ngh(self, x: int):
        return [x] + list(self.graph.neighbors(x))
    
    def scnd_ngh(self, x: int):
        nghs = [list(self.graph.neighbors(first)) for first in list(self.graph.neighbors(x))]
        res = [i for l in nghs for i in l if i != x]
        return list(set(res))
    
    def one_hot(self, arr):
        res = torch.zeros(self.n)
        res[arr] = 1
        return res
        
    
    def vanilla_eq(self):
        pass
    
if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from(list(range(5)))
    edges = [(0,2),(0,1),(1,3),(2,3),(3,4)] 
    G.add_edges_from(edges) 
    
    model = Net(G)
    print(model(torch.ones(len(G.nodes))))
    model.trainable_params()

