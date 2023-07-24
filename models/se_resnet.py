import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class SEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=8):
        """
        in_channel - size of input channel
        out_channel - size of output channel
        stride - stride size of this block. this is applied to first convolution.
        """
        super(SEResBlock, self).__init__()

        self.residual_path = None
        if in_channels != out_channels or stride > 1:

            self.residual_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        # convolution path
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            self.relu,
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # squeeze and excitation path
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitate= nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction),
            self.relu,
            nn.Linear(out_channels // reduction, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        residual - same as x for short-cut connection
        out - feature map which is enhanced by SE
        se_vec - vector that contains SE info
        """
        residual = x
        
        
        out = self.conv_path(x)
        b, c, _, _ = out.size()
        
        se_vec = self.squeeze(out).view(b, c)
        se_vec = self.excitate(se_vec).view(b, c, 1, 1)

        out = out * se_vec
        #print(out.shape)
        if self.residual_path is not None:
            residual = self.residual_path(residual)
        #print(residual.shape)    
        out += residual
        out = self.relu(out)
        
        return out


class SEResNet34(nn.Module):
    """
    """

    def __init__(self, exp_config = config.ExpConfig()):
        """ -> encoders -> decoders -> tconv -> conv2 -> extractors -> fc & pooling
        """
        super(SEResNet34, self).__init__()

        self.instancenorm   = nn.InstanceNorm1d(exp_config.n_mels)

        #       conv1       #
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16 , kernel_size=(7, 7), stride=(2, 1), padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            SEResBlock(in_channels=16, out_channels=16),
            SEResBlock(in_channels=16, out_channels=16),
            SEResBlock(in_channels=16, out_channels=16)
        )
        
        self.layer2 = nn.Sequential(
            SEResBlock(in_channels=16, out_channels=32, stride=2),
            SEResBlock(in_channels=32, out_channels=32),
            SEResBlock(in_channels=32, out_channels=32),
            SEResBlock(in_channels=32, out_channels=32)
        )
        
        self.layer3 = nn.Sequential(
            SEResBlock(in_channels=32, out_channels=64, stride=2),
            SEResBlock(in_channels=64, out_channels=64),
            SEResBlock(in_channels=64, out_channels=64),
            SEResBlock(in_channels=64, out_channels=64),
            SEResBlock(in_channels=64, out_channels=64),
            SEResBlock(in_channels=64, out_channels=64)
        )
        
        self.layer4 = nn.Sequential(
            SEResBlock(in_channels=64, out_channels=128),
            SEResBlock(in_channels=128, out_channels=128),
            SEResBlock(in_channels=128, out_channels=128)
        )
        
        self.sap_linear = nn.Linear(128, 128)
        self.attention = nn.Parameter(torch.FloatTensor(128, 1))
        nn.init.xavier_normal_(self.attention)
        
        self.fc = nn.Linear(128, exp_config.embedding_size)

    def forward(self, x):
        
        x = self.instancenorm(x).unsqueeze(1).detach()
        # conv
        x = self.conv1(x)
        #Encoder Path
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # pooling, embedding
        
        x = torch.mean(x, dim=2, keepdim=True)
        
        x = x.permute(0,3,1,2).squeeze(-1)
        # x.shape == [b, 128, 128, 1]
        h = torch.tanh(self.sap_linear(x))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        x = torch.sum(x * w, dim=1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        
        return x
    