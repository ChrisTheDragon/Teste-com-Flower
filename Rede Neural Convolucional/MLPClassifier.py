from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.faltten = nn.Flatten()
        
        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        v = self.faltten(x)
        return self.layers(v)