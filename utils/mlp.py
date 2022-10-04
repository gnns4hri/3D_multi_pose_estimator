from torch import nn

class PoseEstimatorMLP(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()
        print('MLP input size', input_dimensions)
        negative_slope = 0.1
        self.layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_dimensions, 3072),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(3072, 3072),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(3072, 2048),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(1024, output_dimensions),
        )

    def forward(self, x):
        return self.layers(x)


