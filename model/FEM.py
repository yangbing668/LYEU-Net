import torch
import torch.nn as nn


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FEM, self).__init__()
        self.in_planes = in_planes
        self.fea_space = None
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, 1)
        )

        self.out1 = nn.Sequential(
            nn.Conv2d(in_planes * 2, in_planes * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_planes * 2),
            nn.ReLU(),
            nn.Conv2d(in_planes * 2, out_planes, 1),
        )

        self.out2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_planes * 2),
            nn.ReLU(),
            nn.Conv2d(in_planes * 2, out_planes, 1),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        self.fea_space = nn.Parameter(torch.ones(1, self.in_planes, h, w), requires_grad=True).to('cuda')
        x1 = x * self.conv(self.fea_space)
        x1 = torch.cat([x1, x], dim=1)
        out = self.out1(x1) + self.out2(x)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 8, 128, 128).to('cuda')
    block = FEM(in_planes=8, out_planes=1).to('cuda')
    print(input.size())
    output = block(input)
    print(output.size())
