from torch import nn

class PoseEstimatorMLP(nn.Module):
    def __init__(self, input_dimensions, output_dimensions, splits):
        super().__init__()
        print('MLP input size', input_dimensions)
        negative_slope = 0.1
        self.splits = splits
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dimensions//splits, 8192//splits),  nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(8192//splits,             8192//splits),  nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(8192//splits,             1024),          nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.merged = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*splits,   4096),                     nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(4096,          4096),                     nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(4096,          2048),                     nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(2048,          2048),                     nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(2048,          1024),                     nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(1024,          output_dimensions),
        )


    def forward(self, x):
        nn.Flatten(),
        batch = x.shape[0]
        #print(f"1 {x.shape=}")
        h = self.shared(x.view(batch*self.splits, -1))
        #print(f"3 {h.shape=}")
        h = h.view(batch, -1)
        #print(f"5 {h.shape=}")
        return self.merged(h)

