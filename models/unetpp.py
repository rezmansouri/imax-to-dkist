import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetPP(nn.Module):
    def __init__(self, n_channels, output_ch, starting_ch=32, level=5, deep_supervision=False, **kwargs):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.channels = [starting_ch * 2 ** (i+1) for i in range(level)]
        self.n_channels = n_channels
        self.output_ch = output_ch
        self.level = level

        if level == 5:
            self._init_level_5()
        elif level == 4:
            self._init_level_4()
        elif level == 3:
            self._init_level_3()
        else:
            raise ValueError('Wrong UNet++ Config')

    def _init_level_3(self):
        self.conv0_0 = DoubleConv(
            self.n_channels, self.channels[0], self.channels[0])
        self.conv1_0 = DoubleConv(
            self.channels[0], self.channels[1], self.channels[1])
        self.conv2_0 = DoubleConv(
            self.channels[1], self.channels[2], self.channels[2])

        self.conv0_1 = DoubleConv(
            self.channels[0]+self.channels[1], self.channels[0], self.channels[0])
        self.conv1_1 = DoubleConv(
            self.channels[1]+self.channels[2], self.channels[1], self.channels[1])

        self.conv0_2 = DoubleConv(
            self.channels[0]*2+self.channels[1], self.channels[0], self.channels[0])

        self.up1_0 = nn.ConvTranspose2d(
            self.channels[1], self.channels[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(
            self.channels[2], self.channels[2], 2, stride=2)

        self.up1_1 = nn.ConvTranspose2d(
            self.channels[1], self.channels[1], 2, stride=2)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
            self.final2 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)

    def _init_level_4(self):
        self.conv0_0 = DoubleConv(
            self.n_channels, self.channels[0], self.channels[0])
        self.conv1_0 = DoubleConv(
            self.channels[0], self.channels[1], self.channels[1])
        self.conv2_0 = DoubleConv(
            self.channels[1], self.channels[2], self.channels[2])
        self.conv3_0 = DoubleConv(
            self.channels[2], self.channels[3], self.channels[3])

        self.conv0_1 = DoubleConv(
            self.channels[0]+self.channels[1], self.channels[0], self.channels[0])
        self.conv1_1 = DoubleConv(
            self.channels[1]+self.channels[2], self.channels[1], self.channels[1])
        self.conv2_1 = DoubleConv(
            self.channels[2]+self.channels[3], self.channels[2], self.channels[2])

        self.conv0_2 = DoubleConv(
            self.channels[0]*2+self.channels[1], self.channels[0], self.channels[0])
        self.conv1_2 = DoubleConv(
            self.channels[1]*2+self.channels[2], self.channels[1], self.channels[1])

        self.conv0_3 = DoubleConv(
            self.channels[0]*3+self.channels[1], self.channels[0], self.channels[0])

        self.up1_0 = nn.ConvTranspose2d(
            self.channels[1], self.channels[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(
            self.channels[2], self.channels[2], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(
            self.channels[3], self.channels[3], 2, stride=2)

        self.up1_1 = nn.ConvTranspose2d(
            self.channels[1], self.channels[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(
            self.channels[2], self.channels[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(
            self.channels[3], self.channels[3], 2, stride=2)

        self.up1_2 = nn.ConvTranspose2d(
            self.channels[1],  self.channels[1], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(
            self.channels[2], self.channels[2], 2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(
            self.channels[1], self.channels[1], 2, stride=2)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
            self.final2 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
            self.final3 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)

    def _init_level_5(self):
        self.conv0_0 = DoubleConv(
            self.n_channels, self.channels[0], self.channels[0])
        self.conv1_0 = DoubleConv(
            self.channels[0], self.channels[1], self.channels[1])
        self.conv2_0 = DoubleConv(
            self.channels[1], self.channels[2], self.channels[2])
        self.conv3_0 = DoubleConv(
            self.channels[2], self.channels[3], self.channels[3])
        self.conv4_0 = DoubleConv(
            self.channels[3], self.channels[4], self.channels[4])

        self.conv0_1 = DoubleConv(
            self.channels[0]+self.channels[1], self.channels[0], self.channels[0])
        self.conv1_1 = DoubleConv(
            self.channels[1]+self.channels[2], self.channels[1], self.channels[1])
        self.conv2_1 = DoubleConv(
            self.channels[2]+self.channels[3], self.channels[2], self.channels[2])
        self.conv3_1 = DoubleConv(
            self.channels[3]+self.channels[4], self.channels[3], self.channels[3])

        self.conv0_2 = DoubleConv(
            self.channels[0]*2+self.channels[1], self.channels[0], self.channels[0])
        self.conv1_2 = DoubleConv(
            self.channels[1]*2+self.channels[2], self.channels[1], self.channels[1])
        self.conv2_2 = DoubleConv(
            self.channels[2]*2+self.channels[3], self.channels[2], self.channels[2])

        self.conv0_3 = DoubleConv(
            self.channels[0]*3+self.channels[1], self.channels[0], self.channels[0])
        self.conv1_3 = DoubleConv(
            self.channels[1]*3+self.channels[2], self.channels[1], self.channels[1])

        self.conv0_4 = DoubleConv(
            self.channels[0]*4+self.channels[1], self.channels[0], self.channels[0])

        self.up1_0 = nn.ConvTranspose2d(
            self.channels[1], self.channels[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(
            self.channels[2], self.channels[2], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(
            self.channels[3], self.channels[3], 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(
            self.channels[4], self.channels[4], 2, stride=2)

        self.up1_1 = nn.ConvTranspose2d(
            self.channels[1], self.channels[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(
            self.channels[2], self.channels[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(
            self.channels[3], self.channels[3], 2, stride=2)

        self.up1_2 = nn.ConvTranspose2d(
            self.channels[1],  self.channels[1], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(
            self.channels[2], self.channels[2], 2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(
            self.channels[1], self.channels[1], 2, stride=2)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
            self.final2 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
            self.final3 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
            self.final4 = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(
                self.channels[0], self.output_ch, kernel_size=1)

    def _forward_level_5(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        # up

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))  # up
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))  # up

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return torch.mean(torch.cat([output1, output2, output3, output4], dim=1), dim=1, keepdim=True)

        else:
            output = self.final(x0_4)
            return output

    def _forward_level_4(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        # up

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))  # up
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))  # up

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return torch.mean(torch.cat([output1, output2, output3], dim=1), dim=1, keepdim=True)

        else:
            output = self.final(x0_3)
            return output

    def _forward_level_3(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        # up

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))  # up
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))  # up

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            return torch.mean(torch.cat([output1, output2], dim=1), dim=1, keepdim=True)

        else:
            output = self.final(x0_2)
            return output

    def forward(self, x):
        if self.level == 5:
            return self._forward_level_5(x)
        elif self.level == 4:
            return self._forward_level_4(x)
        elif self.level == 3:
            return self._forward_level_3(x)
