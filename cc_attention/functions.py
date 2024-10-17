'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        # print("x:", x.shape, x[0])
        # torch.Size([32, 32, 50, 545])
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        # print("proj_query:", proj_query.shape, proj_query[0])
        # torch.Size([32, 4, 50, 545])
        # print("proj_query_H:", proj_query_H.shape, proj_query_H[0])
        # torch.Size([17440, 50, 4])
        # print("proj_query_W:", proj_query_W.shape, proj_query_W[0])
        # torch.Size([1600, 545, 4])

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        # print("proj_key:", proj_key.shape, proj_key[0])
        # torch.Size([32, 4, 50, 545])
        # print("proj_key_H:", proj_key_H.shape, proj_key_H[0])
        # torch.Size([17440, 4, 50])
        # print("proj_key_W:", proj_key_W.shape, proj_key_W[0])
        # torch.Size([1600, 4, 545])

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        # print("proj_value:", proj_value.shape, proj_value[0])
        # torch.Size([32, 32, 50, 545])
        # print("proj_value_H:", proj_value_H.shape, proj_value_H[0])
        # torch.Size([17440, 32, 50])
        # print("proj_value_W:", proj_value_W.shape, proj_value_W[0])
        # torch.Size([1600, 32, 545])

        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        # print("energy_H:", energy_H.shape, energy_H[0])
        # torch.Size([32, 50, 545, 50])
        # print("energy_W:", energy_W.shape, energy_W[0])
        # torch.Size([32, 50, 545, 545])

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # print("concate:", concate.shape, concate[0])
        # torch.Size([32, 50, 545, 595])

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        # print("att_H:", att_H.shape, att_H[0])
        # torch.Size([17440, 50, 50])
        # print("att_W:", att_W.shape, att_W[0])
        # torch.Size([1600, 545, 545])
        # print("out_H:", out_H.shape, out_H[0])
        # torch.Size([32, 32, 50, 545])
        # print("out_W:", out_W.shape, out_W[0])
        # torch.Size([32, 32, 50, 545])

        return self.gamma*(out_H + out_W) + x


if __name__ == '__main__':
    model = CrissCrossAttention(64).cuda()
    x = torch.randn(2, 64, 5, 6).cuda()
    out = model(x)
    print(out.shape)
