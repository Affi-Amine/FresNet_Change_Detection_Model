import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, chunk_size=4096):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.chunk_size = chunk_size

    def forward(self, x):
        batch, channels, height, width = x.size()
        pixels = height * width
        
        # Compute query, key, value
        query = self.query_conv(x).view(batch, -1, pixels).permute(0, 2, 1)  # [B, H*W, C//8]
        key = self.key_conv(x).view(batch, -1, pixels)  # [B, C//8, H*W]
        value = self.value_conv(x).view(batch, -1, pixels)  # [B, C, H*W]
        
        # Process attention in chunks
        out = torch.zeros_like(value)  # [B, C, H*W]
        for i in range(0, pixels, self.chunk_size):
            end = min(i + self.chunk_size, pixels)
            query_chunk = query[:, i:end, :]  # [B, chunk_size, C//8]
            energy = torch.bmm(query_chunk, key)  # [B, chunk_size, H*W]
            attention = self.softmax(energy)  # [B, chunk_size, H*W]
            out[:, :, i:end] = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, chunk_size]
        
        out = out.view(batch, channels, height, width)
        return self.gamma * out + x

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, t):
        return self.linear(t.unsqueeze(-1))

class BasicBlock_ss(nn.Module):
    def __init__(self, inplanes, planes=None, subsamp=1):
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
        
        out += residual
        out = self.relu(out)
        return out

class BasicBlock_us(nn.Module):
    def __init__(self, inplanes, upsamp=1):
        super(BasicBlock_us, self).__init__()
        planes = int(inplanes / upsamp)
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsamp = upsamp
        self.couple = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1)
        self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.couple(x)
        residual = self.bnc(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class FresUNet(nn.Module):
    def __init__(self, input_nbr=6, label_nbr=2):
        super(FresUNet, self).__init__()
        self.input_nbr = input_nbr
        base_depth = 8

        self.time_embed = TimeEmbedding(embed_dim=64)
        self.time_conv = nn.Conv2d(64, input_nbr, kernel_size=1)

        self.encres1_1 = BasicBlock_ss(input_nbr, planes=base_depth)
        cur_depth = base_depth
        d1 = base_depth
        self.encres1_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2

        self.encres2_1 = BasicBlock_ss(cur_depth)
        d2 = cur_depth
        self.encres2_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2

        self.encres3_1 = BasicBlock_ss(cur_depth)
        d3 = cur_depth
        self.encres3_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2

        self.encres4_1 = BasicBlock_ss(cur_depth)
        d4 = cur_depth
        self.encres4_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2

        self.attn4 = SelfAttention(d4)

        self.decres4_1 = BasicBlock_ss(cur_depth)
        self.decres4_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth/2)

        self.decres3_1 = BasicBlock_ss(cur_depth + d4, planes=cur_depth)
        self.decres3_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth/2)

        self.decres2_1 = BasicBlock_ss(cur_depth + d3, planes=cur_depth)
        self.decres2_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth/2)

        self.decres1_1 = BasicBlock_ss(cur_depth + d2, planes=cur_depth)  # Output 16 channels
        self.attn1 = SelfAttention(cur_depth)  # Expect 16 channels
        self.decres1_2 = BasicBlock_us(cur_depth, upsamp=2)  # Output 8 channels
        cur_depth = int(cur_depth/2)  # 8 channels

        self.coupling = nn.Conv2d(cur_depth + d1, label_nbr, kernel_size=1)
        self.sm = nn.LogSoftmax(dim=1)

        recon_channels = input_nbr // 2 if input_nbr in [6, 26] else 3
        self.recon_conv = nn.Conv2d(cur_depth + d1, recon_channels, kernel_size=1)

    def forward(self, x1, x2, t):
        time_emb = self.time_embed(t)
        batch_size, _, height, width = x1.size()
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        time_emb = self.time_conv(time_emb)
        time_emb = time_emb.expand(-1, -1, height, width)
        
        x = torch.cat((x1, x2), 1) + time_emb
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
        x4 = self.attn4(x4)
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

        x = torch.cat((x, x2), 1)  # 16 + 16 = 32 channels
        x = self.decres1_1(x)  # Output 16 channels
        x = self.attn1(x)  # Input 16 channels
        x = self.decres1_2(x)  # Output 8 channels
        s1_2 = x.size()
        pad1 = ReplicationPad2d((0, s1_1[3] - s1_2[3], 0, s1_1[2] - s1_2[2]))  # Fixed typo
        x = pad1(x)

        recon_input = torch.cat((x, x1), 1)  # 8 + 8 = 16 channels
        change_pred = self.coupling(recon_input)
        change_pred = self.sm(change_pred)
        recon = self.recon_conv(recon_input)
        return change_pred, recon