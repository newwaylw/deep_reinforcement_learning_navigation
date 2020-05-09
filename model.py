import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_states=128, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.h = hidden_states
        
        self.model = torch.nn.Sequential(
        torch.nn.Linear(state_size, self.h),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(self.h, self.h//2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(self.h//2, action_size),
    )
        
        self.model.zero_grad()
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)
