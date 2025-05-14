# Enhanced FresUNet with Attention and Multi-scale Features
# Based on original by Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Added attention mechanisms and enhanced feature extraction for small changes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class BasicBlock_ss(nn.Module):
    def __init__(self, inplanes, planes=None, subsamp=1, use_attention=False):
        super(BasicBlock_ss, self).__init__()
        if planes is None:
            planes = inplanes * subsamp
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.subsamp = subsamp
        self.doit = planes != inplanes
        if self.doit:
            self.couple = nn.Conv2d(inplanes, planes, kernel_size=1)
            self.bnc = nn.BatchNorm2d(planes)
        
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(planes)

    def forward(self, x):
        if self.doit:
            residual = self.couple(x)
            residual = self.bnc(residual)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.subsamp > 1:
            out = F.max_pool2d(out, kernel_size=self.subsamp, stride=self.subsamp)
            residual = F.max_pool2d(residual, kernel_size=self.subsamp, stride=self.subsamp)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.cbam(out)
        
        out += residual
        out = self.relu(out)

        return out
    

class BasicBlock_us(nn.Module):
    def __init__(self, inplanes, upsamp=1, use_attention=False):
        super(BasicBlock_us, self).__init__()
        planes = int(inplanes / upsamp)  # assumes integer result, fix later
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsamp = upsamp
        self.couple = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1) 
        self.bnc = nn.BatchNorm2d(planes)
        
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(planes)

    def forward(self, x):
        residual = self.couple(x)
        residual = self.bnc(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out
    

class AttentionFresUNet(nn.Module):
    """Enhanced FresUNet with attention mechanisms for better small change detection."""

    def __init__(self, input_nbr, label_nbr):
        """Initialize AttentionFresUNet fields."""
        super(AttentionFresUNet, self).__init__()
        
        self.input_nbr = input_nbr
        
        cur_depth = input_nbr
        
        # Increased base depth for better feature extraction
        base_depth = 8
        
        # Encoding stage 1
        self.encres1_1 = BasicBlock_ss(cur_depth, planes=base_depth)
        cur_depth = base_depth
        d1 = base_depth
        self.encres1_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 2
        self.encres2_1 = BasicBlock_ss(cur_depth, use_attention=False)  # Removed attention
        d2 = cur_depth
        self.encres2_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 3
        self.encres3_1 = BasicBlock_ss(cur_depth, use_attention=True)  # Keep attention in middle layers
        d3 = cur_depth
        self.encres3_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 4
        self.encres4_1 = BasicBlock_ss(cur_depth, use_attention=True)
        d4 = cur_depth
        self.encres4_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Decoding stage 4
        self.decres4_1 = BasicBlock_ss(cur_depth, use_attention=True)
        self.decres4_2 = BasicBlock_us(cur_depth, upsamp=2, use_attention=False)  # Removed attention
        cur_depth = int(cur_depth/2)
        
        # Decoding stage 3
        self.decres3_1 = BasicBlock_ss(cur_depth + d4, planes=cur_depth, use_attention=True)
        self.decres3_2 = BasicBlock_us(cur_depth, upsamp=2, use_attention=False)  # Removed attention
        cur_depth = int(cur_depth/2)
        
        # Decoding stage 2
        self.decres2_1 = BasicBlock_ss(cur_depth + d3, planes=cur_depth, use_attention=False)  # Removed attention
        self.decres2_2 = BasicBlock_us(cur_depth, upsamp=2, use_attention=False)  # Removed attention
        cur_depth = int(cur_depth/2)
        
        # Decoding stage 1
        self.decres1_1 = BasicBlock_ss(cur_depth + d2, planes=cur_depth)
        self.decres1_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth/2)
        
        # Multi-scale feature fusion
        self.scale4_proj = nn.Conv2d(d4, 32, kernel_size=1)
        self.scale3_proj = nn.Conv2d(d3, 32, kernel_size=1)
        self.scale2_proj = nn.Conv2d(d2, 32, kernel_size=1)
        self.scale1_proj = nn.Conv2d(d1, 32, kernel_size=1)
        
        # Output
        self.fusion = nn.Conv2d(32*4 + cur_depth + d1, 16, kernel_size=3, padding=1)
        self.fusion_bn = nn.BatchNorm2d(16)
        self.final_attention = CBAM(16)
        self.coupling = nn.Conv2d(16, label_nbr, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(label_nbr)
        # REMOVE or comment out this line:
        # self.sm = nn.LogSoftmax(dim=1)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        
        s1_1 = x.size()
        x1 = self.encres1_1(x)
        x = self.encres1_2(x1)
        
        s2_1 = x.size()
        x2 = self.encres2_1(x)
        x = self.encres2_2(x2)
        
        s3_1 = x.size()
        x3 = self.encres3_1(x)
        x = self.encres3_2(x3)
        
        s4_1 = x.size()
        x4 = self.encres4_1(x)
        x = self.encres4_2(x4)
        
        x = self.decres4_1(x)
        x = self.decres4_2(x)
        s4_2 = x.size()
        pad4 = ReplicationPad2d((0, s4_1[3] - s4_2[3], 0, s4_1[2] - s4_2[2]))
        x = pad4(x)
        
        x = self.decres3_1(torch.cat((x, x4), 1))
        x = self.decres3_2(x)
        s3_2 = x.size()
        pad3 = ReplicationPad2d((0, s3_1[3] - s3_2[3], 0, s3_1[2] - s3_2[2]))
        x = pad3(x)
        
        x = self.decres2_1(torch.cat((x, x3), 1))
        x = self.decres2_2(x)
        s2_2 = x.size()
        pad2 = ReplicationPad2d((0, s2_1[3] - s2_2[3], 0, s2_1[2] - s2_2[2]))
        x = pad2(x)
        
        x = self.decres1_1(torch.cat((x, x2), 1))
        x = self.decres1_2(x)
        s1_2 = x.size()
        pad1 = ReplicationPad2d((0, s1_1[3] - s1_2[3], 0, s1_1[2] - s1_2[2]))
        x = pad1(x)
        
        # Multi-scale feature fusion
        f4 = self.scale4_proj(F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=False))
        f3 = self.scale3_proj(F.interpolate(x3, size=x.size()[2:], mode='bilinear', align_corners=False))
        f2 = self.scale2_proj(F.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=False))
        f1 = self.scale1_proj(x1)
        
        # Concatenate all features for final prediction
        multi_scale = torch.cat([f1, f2, f3, f4, x, x1], dim=1)
        
        fused = self.fusion(multi_scale)
        fused = self.fusion_bn(fused)
        fused = F.relu(fused)
        fused = self.final_attention(fused)
        
        out = self.coupling(fused)
        out = self.final_bn(out)
        # REMOVE temperature scaling and LogSoftmax
        return out
        # Remove the duplicate return statement