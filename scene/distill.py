import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, y):
        x = self.query(x)
        y = self.key(y)
        attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        V = self.value(y)
        output = torch.bmm(attn_weights, V)
        
        return output
    
class DistillModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = nn.Linear(self.in_dim*2, self.out_dim) 
        torch.nn.init.normal_(self.layer.weight, mean=0.5, std=1)
        self.act = nn.ReLU()
        # self.drop = nn.Dropout(0.15, inplace=True)
        self.pool_filter = nn.MaxPool1d(kernel_size=3, return_indices=True)
        self.threshold = 0.5
        
    def forward(self, src, ref, opa=None):
        # Input src, ref: [N, 3]
        # in_feat = torch.cat([src, ref], dim=1).permute(1,0).unsqueeze(0)
        in_feat = torch.cat([src, ref], dim=1).unsqueeze(0)
        
        in_feat = self.layer(in_feat)+0.5
        # print(in_feat.max().item())
        # print(in_feat.min().item())
        # print(in_feat.shape)
        # print(in_feat)
        
        in_feat = self.act(in_feat)
        
        # print(in_feat.max().item())
        # print(in_feat.min().item())
        # in_feat = in_feat / in_feat.max()
        
        # print(in_feat.max().item())
        # print(in_feat)
        # exit()
        # in_feat = in_feat # - src
        # in_feat = in_feat > self.threshold
        # print(in_feat.shape)
        
        in_feat = in_feat.squeeze()
        # print(in_feat.shape)
        
        # in_feat = self.drop(self.act(in_feat) * ref.permute(1,0).unsqueeze(0))
        # in_feat = src + in_feat.squeeze(0).permute(1,0)
        return in_feat
    
    
class PoolingModule(nn.Module):
    def __init__(self, pool_list):
        super().__init__()
        self.pool_list = pool_list
        self.filter_list = []
        for i in self.pool_list:
            pool_filter = nn.MaxPool1d(kernel_size=i, return_indices=True)
            self.filter_list.append(pool_filter)
                
    def forward(self, mean, scale, rot, opa, shs):
        # Input mean: [N, 3]
        mean_ori, scale_ori, rot_ori, opa_ori, shs_ori = \
            mean, scale, rot, opa, shs 
        result_dict = {'mean':[], 'scale':[], 'rot':[], 'opa':[], 'shs':[]}
        for i in self.filter_list:
            opa, pool_idx = i(opa_ori[None].permute(0,2,1))
            pool_idx = pool_idx.squeeze()
            result_dict['opa'].append(opa.permute(0,2,1).squeeze(0))
            result_dict['mean'].append(mean_ori[pool_idx])
            result_dict['scale'].append(scale_ori[pool_idx])
            result_dict['rot'].append(rot_ori[pool_idx])
            result_dict['shs'].append(shs_ori[pool_idx])
            
        return result_dict
        