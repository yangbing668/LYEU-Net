import torch
import torch.nn as nn


class MBM(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MBM, self).__init__()

        self.branch_x = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.GroupNorm(4, dim_out),
            nn.GELU(),
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in // 4, kernel_size=3, padding=1, dilation=1, groups=dim_in // 4),
            nn.GroupNorm(4, dim_in // 4),
            nn.GELU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in // 4, kernel_size=3, padding=2, dilation=2, groups=dim_in // 4),
            nn.GroupNorm(4, dim_in // 4),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in // 4, kernel_size=3, padding=3, dilation=3, groups=dim_in // 4),
            nn.GroupNorm(4, dim_in // 4),
            nn.GELU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in // 4, kernel_size=3, padding=4, dilation=4, groups=dim_in // 4),
            nn.GroupNorm(4, dim_in // 4),
            nn.GELU(),
        )

        # Attention模块
        self.attention1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in // 4, dim_in // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in // 8, dim_in // 4, kernel_size=1),
            nn.Sigmoid()
        )

        # 输出卷积层
        self.out_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        feature1 = self.branch1(x)
        feature1 = self.attention1(feature1) * feature1
        feature2 = self.branch2(x)
        feature2 = self.attention1(feature2) * feature2
        feature3 = self.branch3(x)
        feature3 = self.attention1(feature3) * feature3
        feature4 = self.branch4(x)
        feature4 = self.attention1(feature4) * feature4

        # 自适应融合
        fused_feature = torch.cat([feature1, feature2, feature3, feature4], dim=1)

        # 输出处理
        output = self.out_conv(fused_feature) + self.branch_x(x)

        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.rand(1, 16, 256, 256)
    attention_module = MBM(16, 32)
    output_tensor = attention_module(input_tensor)
    # 打印输入和输出的形状
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
