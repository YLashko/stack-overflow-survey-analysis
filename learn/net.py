import torch
from torch import nn

class SalaryModelLinear(nn.Module):
    def __init__(self,
            input_shape: int,
            hidden_units: int,
            output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, X):
        return self.layer_stack(X)

class SalaryModelLinearDoubleReLU(nn.Module):
    def __init__(self,
            input_shape: int,
            hidden_units: int,
            output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )
    
    def forward(self, X):
        return self.layer_stack(X)

loss_fn = nn.SmoothL1Loss()

def get_optim(parameters, lr):
    return torch.optim.Adam(params=parameters, lr=lr)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: nn.Module, loss_fn: nn.L1Loss, optimizer: torch.optim.SGD, x, y, epoch):
    model.train()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test_step(model: nn.Module, loss_fn: nn.L1Loss, optimizer: torch.optim.SGD, x, y, epoch):
    model.eval()
    with torch.inference_mode():
        test_y_pred = model(x)
        test_loss = loss_fn(test_y_pred, y)
        print(f"Epoch {epoch} | Test loss: {test_loss} | Test std {test_y_pred.squeeze().std()}")
