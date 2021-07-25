import torch
import torch.nn as nn


class ResNet(nn.Module):
    r"""Resnet layers implementation summarized in the paper (layer 2~3)"""

    def __init__(self, in_feature, hid_feature, out_feature, repeat=8, k_size=5, stride=1, pad=2):
        super(ResNet, self).__init__()
        # layer 1
        self.layer_top = nn.Sequential(
            nn.Conv2d(in_feature, hid_feature, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(hid_feature),
            nn.ReLU(inplace=True), )
        # layer 2~17
        self.layer_middles = nn.ModuleList()
        for i in range(repeat):
            self.layer_middles.append(nn.Sequential(
                # layer 2
                nn.Conv2d(hid_feature, hid_feature, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hid_feature),
                nn.ReLU(inplace=True),
                # layer 3
                nn.Conv2d(hid_feature, hid_feature, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hid_feature),
                nn.ReLU(inplace=True), ))
        # layer 18
        self.layer_bottom = nn.Conv2d(hid_feature, out_feature, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.layer_top(x)
        for layer in self.layer_middles:
            x = layer(x) + x
        x = self.layer_bottom(x)
        return x


class MCDLNet(nn.Module):
    def __init__(self, n=32, d=160):
        super(MCDLNet, self).__init__()
        self.n = n
        self.d = d
        # Weight volume generation
        # layer 1~18
        self.resnet1 = ResNet(1, n, n, stride=2)

        # layer 19~20
        self.attention = nn.Sequential(
            nn.Conv3d(2 * n, n, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv3d(n, 1, kernel_size=1),
            nn.Sigmoid(), )

        self.regulation = nn.ModuleList()
        # layer 21~22
        self.regulation.append(nn.Sequential(
            nn.Conv3d(2 * n, n, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(n),
            nn.ReLU(inplace=True),
            nn.Conv3d(n, n, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(n),
            nn.ReLU(inplace=True), ))
        # layer 23~25, 26~28, 29~31, 32~34
        for i in range(4):
            self.regulation.append(nn.Sequential(
                nn.Conv3d(n if i == 0 else 2 * n, 2 * n, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(2 * n),
                nn.ReLU(inplace=True),
                nn.Conv3d(2 * n, 2 * n, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(2 * n),
                nn.ReLU(inplace=True),
                nn.Conv3d(2 * n, 2 * n, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(2 * n),
                nn.ReLU(inplace=True), ))
        # layer 35, 36, 37, 38
        for i in range(4):
            self.regulation.append(nn.Sequential(
                nn.ConvTranspose3d(2 * n, n if i == 3 else 2 * n, kernel_size=3, stride=2, padding=0, bias=False),
                nn.BatchNorm3d(n if i == 3 else 2 * n),
                nn.ReLU(inplace=True), ))
        # layer 39
        self.regulation.append(nn.ConvTranspose3d(n, 1, kernel_size=3, stride=2, padding=0, bias=False))

        # Color residue joint learning
        self.resnet2 = ResNet(1, n, n)
        self.resnet3 = ResNet(1, n, n)
        self.resnet4 = ResNet(2 * n, n, 1)

    def forward(self, mono_img, color_img_y, color_img_c):
        # Weight volume generation
        F_Y = self.resnet1(mono_img)
        F_YR = self.resnet1(color_img_y)

        V_FR = torch.zeros(F_Y.shape + (self.d // 2,), device=F_Y.device)  # NCHWD
        for i in range(self.d // 2):
            V_FR[:, :, :, i:, i] = F_YR if i == 0 else F_YR[:, :, :, :-i]  # grad_fn=<CopySlices>
        V_F = torch.cat([F_Y.unsqueeze(dim=-1).expand_as(V_FR), V_FR], dim=1)

        A = self.attention(V_F)
        V_A = torch.cat([V_F[:, :self.n], V_F[:, self.n:] * A], dim=1)

        W_22 = self.regulation[0](V_A)  # torch.Size([1, 32, 128, 256, 80])
        W_25 = self.regulation[1](W_22)  # torch.Size([1, 64, 64, 128, 40])
        W_28 = self.regulation[2](W_25)  # torch.Size([1, 64, 32, 64, 20])
        W_31 = self.regulation[3](W_28)  # torch.Size([1, 64, 16, 32, 10])
        W_34 = self.regulation[4](W_31)  # torch.Size([1, 64, 8, 16, 5])
        W_35 = self.regulation[5](W_34)[:, :, :-1, :-1, :-1] + W_31  # torch.Size([1, 64, 16, 32, 10])
        W_36 = self.regulation[6](W_35)[:, :, :-1, :-1, :-1] + W_28  # torch.Size([1, 64, 32, 64, 20])
        W_37 = self.regulation[7](W_36)[:, :, :-1, :-1, :-1] + W_25  # torch.Size([1, 64, 64, 128, 40])
        W_38 = self.regulation[8](W_37)[:, :, :-1, :-1, :-1] + W_22  # torch.Size([1, 32, 128, 256, 80])
        W = self.regulation[9](W_38)[:, :, :-1, :-1, :-1]  # torch.Size([1, 1, 256, 512, 160])

        # Color residue joint learning
        C_R = color_img_c
        C_P = torch.zeros(C_R.shape + (self.d,), device=C_R.device)  # NCHWD
        for i in range(self.d):
            C_P[:, :, :, i:, i] = C_R if i == 0 else C_R[:, :, :, :-i]
        C_P = (C_P * W).sum(dim=-1)

        G_CP = self.resnet2(C_P)
        G_Y = self.resnet3(mono_img)
        G = torch.cat([G_CP, G_Y], dim=1)
        C = self.resnet4(G)
        #return C + C_P
        return C  # non-residual shows more generalized results.


if __name__ == "__main__":
    zeros1 = torch.zeros([4, 1, 256, 512])
    resnet1 = ResNet(1, 32, 32, k_size=5, stride=2, pad=2)
    print("resnet1", resnet1(zeros1).shape)
    resnet2 = ResNet(1, 32, 32)
    print("resnet2", resnet2(zeros1).shape)
    mcdlnet = MCDLNet().cuda()
    mono_img = torch.zeros([1, 1, 256 // 2, 512 // 2]).cuda()
    color_img = torch.zeros([1, 3, 256 // 2, 512 // 2]).cuda()
    mcdlnet1 = mcdlnet(mono_img, color_img[:, 0:1], color_img[:, 1:2])
    print("mcdlnet", mcdlnet1.shape)
