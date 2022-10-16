import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from collections import OrderedDict
from typing import List

class LinearArcsine(nn.Linear):
    """
    Linear layer with arcsine activation:
    x -> arcsine(normalize([x,1]).T @ normalize([W;b]))
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearArcsine, self).__init__(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((1,out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            W = torch.concat([self.weight, self.bias.T], dim = 1) # [out_features,in_features+1]
            input = torch.concat([input, torch.ones((input.shape[0],1),device=input.device)], dim=1) #　[batch_size, in_features+1]
        else:
            W = self.weight
        return torch.asin(F.normalize(input) @ F.normalize(W).T)

class RandomFeatureMap(nn.Module):        
    """
    Random feature map for arcsin
    """
    def __init__(self, in_features: int, out_features: int = None, seed: int = None, device = 'cpu', indicator = lambda x : (x > 0).float()):
        """
        Build a feature map for arcsin
        indicator: a lambda function that is either the indicator function, or a sigmoid function
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        """
        weights is d by d Gaussian matrix 
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        weights = [nn.Parameter(torch.tensor(rng.normal(loc=0.0, scale=1.0, size=(self.in_features,self.out_features)),device=device,dtype=torch.float32), requires_grad=False)] # for each N, generate N random Rademacher vectors
        self.weights = nn.ParameterList(weights)

        self.indicator = indicator
    
    
    def forward(self, x):
        """
        return feature map for arcsin: for each Z in weights matrix have the feature sign(Z dot x) = 2*indicator(Z dot x) -1
        with these features, phix^T phiy estimates d*(2/pi)arcsin(x^T y / ||x||||y||) 
        """

        return torch.stack([2*self.indicator(x @ weight)-1 for weight in self.weights]).squeeze(0)


class ArcsinNN(nn.Module):
    """
    arcsin
    nn with arcsine activation
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: List[int] = None, bias: bool = True):
        """
        Initialize an ArcsinNN
        hidden_features: a list contains the dimension of hidden layers
        """
        super(ArcsinNN, self).__init__()
        self.num_hidden_layers = len(hidden_features) if hidden_features else 0
        self.Flatten = nn.Flatten() # flatten the input
        self.has_bias = bias

        if self.num_hidden_layers:
            Layers = []
            dims = [in_features] + hidden_features
            for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
                Layers.append(('LinearArcsine'+f'{i}', LinearArcsine(in_features = in_dim, out_features = out_dim, bias=self.has_bias)))
            Layers.append(('Output', nn.Linear(in_features = hidden_features[-1], out_features= out_features)))
            self.Layers = nn.Sequential(OrderedDict(Layers))
            
        else:
            raise ValueError("Missing hidden_feautres!")
        
    def forward(self, x):
       
        x = self.Flatten(x)
        logits = self.Layers(x)
        
        return logits

class ApproxArcsineNN(nn.Module):
    """
    Given a valid ArcsinNN model, approximate the ArcsinNN using RandomFeatureMap for each LinearArcsine layers
    """
    def __init__(self, model: ArcsinNN = None, indicator = lambda x: (x>0).float(), seed: int = None, project_dim: int = None):
        super(ApproxArcsineNN, self).__init__()
        if model is None or type(model) is not ArcsinNN:
            raise ValueError("Missing input ArcsinNN model!")

        self.num_hidden_layers = model.num_hidden_layers

        self.Flatten = nn.Flatten() # flatten the input
        
        self.has_bias = model.has_bias

        dims = set(model.Layers[i].in_features for i in range(self.num_hidden_layers))

        if self.has_bias:
            # create random feature maps
            self.RandomFeatureMaps = {d: RandomFeatureMap(d+1, out_features=project_dim, device=model.Layers[0].weight.device, indicator=indicator, seed=seed).float() for d in dims}
            
            # copy and paste weights
            self.Linears = nn.ModuleList([nn.Linear(in_features=model.Layers[i].in_features, out_features=model.Layers[i].out_features) for i in range(self.num_hidden_layers)])
            for i in range(self.num_hidden_layers):
                self.Linears[i].weight = nn.Parameter(model.Layers[i].weight.clone().detach())
                self.Linears[i].bias = nn.Parameter(model.Layers[i].bias.clone().detach())
        else:
            self.RandomFeatureMaps = {d: RandomFeatureMap(d, out_features=project_dim, device=model.Layers[0].weight.device, indicator=indicator, seed=seed).float() for d in dims}

            # copy and paste weights
            self.Linears = nn.ModuleList([nn.Linear(in_features=model.Layers[i].in_features, out_features=model.Layers[i].out_features,bias=False) for i in range(self.num_hidden_layers)])
            for i in range(self.num_hidden_layers):
                self.Linears[i].weight = nn.Parameter(model.Layers[i].weight.clone().detach())

        self.Output = nn.Linear(in_features=model.Layers[-1].in_features, out_features=model.Layers[-1].out_features)
        self.Output.weight = nn.Parameter(model.Layers[-1].weight.clone().detach())
        self.Output.bias = nn.Parameter(model.Layers[-1].bias.clone().detach())

    def forward(self, x):

        x = self.Flatten(x) # [n, D_in]
        
        for i in range(self.num_hidden_layers):
            n, D = x.shape[0], x.shape[1]
            if self.has_bias:
                x = torch.concat([x, torch.ones((n, 1), device=x.device)], dim = 1) # [n, D_in + 1]
                W = torch.concat([self.Linears[i].weight, self.Linears[i].bias.T], dim = 1) # [D_out, D_in + 1]
            else:
                W = self.Linears[i].weight # [D_out, D_in]
            phi_x = self.RandomFeatureMaps[D](x) # [n, d_featuremap]
            phi_W = self.RandomFeatureMaps[D](W) # [D_out, d_featuremap]
            if self.has_bias:
                x = (np.pi/2)*((phi_x @ phi_W.T)/(D+1))
            else:
                x = (np.pi/2)*((phi_x @ phi_W.T)/(D))

        # output layer
        logits = self.Output(x)
        
        return logits

class RepresentArcsineNN(nn.Module):
    """
    Given a valid ArcsinNN model, approximate the ArcsinNN using RandomFeatureMap for each LinearArcsine layers, and represent using composition of feature maps.
    """
    def __init__(self, model: ArcsinNN = None, indicator = lambda x: (x > 0).float(), seed:int = None, ignore_first_layer:bool = False, project_dim: int = None):
        super(RepresentArcsineNN, self).__init__()
        if model is None or type(model) is not ArcsinNN:
            raise ValueError("Missing input ArcsinNN model!")

        self.num_hidden_layers = model.num_hidden_layers
        self.has_bias = model.has_bias
        self.ignore_first_layer = ignore_first_layer

        self.Flatten = nn.Flatten() # flatten the input
        
        if self.has_bias:
            # create random feature maps
            if ignore_first_layer:
                self.input_dim = model.Layers[1].in_features
                self.project_dim = project_dim or self.input_dim
                self.RandomFeatureMaps = {d + 1: RandomFeatureMap(d + 1, out_features=self.project_dim, device = model.Layers[0].weight.device, indicator=indicator, seed=seed).float() for d in [self.input_dim, self.project_dim]}
                self.Linears = nn.ModuleList([LinearArcsine(in_features=model.Layers[0].in_features, out_features=model.Layers[0].out_features)] 
                                + [nn.Linear(in_features=model.Layers[i].in_features, out_features=model.Layers[i].out_features) for i in range(1,self.num_hidden_layers)])
                for i in range(self.num_hidden_layers):
                    self.Linears[i].weight = nn.Parameter(model.Layers[i].weight.clone().detach())
                    self.Linears[i].bias = nn.Parameter(model.Layers[i].bias.clone().detach())
            else:
                self.input_dim = model.Layers[0].in_features
                self.project_dim = project_dim or self.input_dim
                self.RandomFeatureMaps = {d + 1: RandomFeatureMap(d + 1, out_features=self.project_dim, device = model.Layers[0].weight.device, indicator=indicator, seed=seed).float() for d in [self.input_dim, self.project_dim]}

                # copy and paste weights
                self.Linears = nn.ModuleList([nn.Linear(in_features=model.Layers[i].in_features, out_features=model.Layers[i].out_features) for i in range(self.num_hidden_layers)])
                for i in range(self.num_hidden_layers):
                    self.Linears[i].weight = nn.Parameter(model.Layers[i].weight.clone().detach())
                    self.Linears[i].bias = nn.Parameter(model.Layers[i].bias.clone().detach())

          
        else:
            if ignore_first_layer:
                self.input_dim = model.Layers[1].in_features
                self.project_dim = project_dim or self.input_dim
                self.Linears = nn.ModuleList([LinearArcsine(in_features=model.Layers[0].in_features, out_features=model.Layers[0].out_features, bias=False)] + 
                                [nn.Linear(in_features=model.Layers[i].in_features, out_features=model.Layers[i].out_features,bias=False) for i in range(1, self.num_hidden_layers)])
                self.RandomFeatureMaps = {d: RandomFeatureMap(d, out_features=self.project_dim, device=model.Layers[0].weight.device, indicator=indicator, seed=seed).float() for d in [self.input_dim, self.project_dim]}

            else:
                self.input_dim = model.Layers[0].in_features
                self.project_dim = project_dim or self.input_dim
                 # copy and paste weights
                self.Linears = nn.ModuleList([nn.Linear(in_features=model.Layers[i].in_features, out_features=model.Layers[i].out_features,bias=False) for i in range(self.num_hidden_layers)])
                for i in range(self.num_hidden_layers):
                    self.Linears[i].weight = nn.Parameter(model.Layers[i].weight.clone().detach())
                self.RandomFeatureMaps = {d: RandomFeatureMap(d, out_features=self.project_dim, device=model.Layers[0].weight.device, indicator=indicator, seed=seed).float() for d in [self.input_dim, self.project_dim]}
        
        self.Output = nn.Linear(in_features=model.Layers[-1].in_features, out_features=model.Layers[-1].out_features)
        self.Output.weight = nn.Parameter(model.Layers[-1].weight.clone().detach())
        self.Output.bias = nn.Parameter(model.Layers[-1].bias.clone().detach())

    def forward(self, x):
        x = self.Flatten(x)
        
        n, D = x.shape[0], x.shape[1]
        if self.has_bias:
            if self.ignore_first_layer:
                x = self.Linears[0](x)
                W = torch.eye(self.Linears[1].weight.shape[1], device=self.Linears[1].weight.device) 
                for i in range(1,self.num_hidden_layers):
                    # compute phi(phi(...phi(x))) 
                    x = torch.concat([x, torch.ones((n, 1), device=x.device)], dim = 1)   
                    x = self.RandomFeatureMaps[x.shape[1]](x)

                    # compute phi(phi(...phi(W)))
                    W = torch.matmul(self.Linears[i].weight, W) # [D_out, D_in]
                    W = torch.concat([W, self.Linears[i].bias.T], dim = 1) # [D_out, D_in + 1]
                    W = self.RandomFeatureMaps[W.shape[1]](W)
                    W = (torch.pi/2/W.shape[1]) * W
            else:
                W = torch.eye(self.Linears[0].weight.shape[1], device=self.Linears[0].weight.device)
                for i in range(self.num_hidden_layers):
                    # compute phi(phi(...phi(x))) 
                    x = torch.concat([x, torch.ones((n, 1), device=x.device)], dim = 1)   
                    x = self.RandomFeatureMaps[x.shape[1]](x)

                    # compute phi(phi(...phi(W)))
                    W = torch.matmul(self.Linears[i].weight, W) # [D_out, D_in]
                    W = torch.concat([W, self.Linears[i].bias.T], dim = 1) # [D_out, D_in + 1]
                    W = self.RandomFeatureMaps[W.shape[1]](W)
                    W = (torch.pi/2/W.shape[1]) * W
        else:
            if self.ignore_first_layer:
                x = self.Linears[0](x)
                W = torch.eye(self.Linears[1].weight.shape[1], device=self.Linears[1].weight.device)
                for i in range(1, self.num_hidden_layers):
                    # compute phi(phi(...phi(x))) 
                    x = self.RandomFeatureMaps[x.shape[1]](x)

                    # compute phi(phi(...phi(W)))
                    W = torch.matmul(self.Linears[i].weight, W)
                    W = self.RandomFeatureMaps[W.shape[1]](W)
                    W = (torch.pi/2/W.shape[1]) * W
            else:
                W = torch.eye(self.Linears[0].weight.shape[1], device=self.Linears[0].weight.device)
                for i in range(self.num_hidden_layers):
                    # compute phi(phi(...phi(x))) 
                    x = self.RandomFeatureMaps[x.shape[1]](x)

                    # compute phi(phi(...phi(W)))
                    W = torch.matmul(self.Linears[i].weight, W)
                    W = self.RandomFeatureMaps[W.shape[1]](W)
                    W = (torch.pi/2/W.shape[1]) * W
        x = torch.matmul(x, W.T)
        # Monitor the property of this W*self.Output.weight.T
        logits = self.Output(x)
        return logits