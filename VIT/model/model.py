import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torch
import einops
from einops.layers.torch import Rearrange

import math
class Attention(BaseModel):
    def __init__(self, num_head, dim):
        super().__init__()
        self.num_head = num_head
        self.dim = dim
        self.to_qkv = nn.Linear(dim, 3*dim*num_head)
        self.from_qkv = nn.Linear(dim*num_head, dim)
    
    def forward(self, x):
        batch = x.shape[0]
        num_patch = x.shape[1]
        qkv = self.to_qkv(x) # (batch, num_patch, dim) -> (batch, num_patch, 3*dim*head)
        
        qkv = einops.rearrange(qkv, 'b n (dim head) -> b head n dim', head=self.num_head, dim=self.dim*3)
        
        q = qkv[:,:,:, 0:self.dim]
        k = qkv[:,:,:, self.dim:2*self.dim]
        v = qkv[:,:,:, 2*self.dim:3*self.dim]

        assert q.shape == torch.Size([batch, self.num_head, num_patch, self.dim])
        assert k.shape == torch.Size([batch, self.num_head, num_patch, self.dim])
        assert v.shape == torch.Size([batch, self.num_head, num_patch, self.dim])

        A = torch.matmul(q, k.transpose(2,3)) * math.sqrt(self.dim)
        A = F.softmax(A)
        
        #print(A.shape)
        #print(v.shape)
        msa = torch.matmul(A, v)
        
        msa = einops.rearrange(msa, 'b head n dim -> b n (head dim)')
        msa = self.from_qkv(msa)
        return msa

class TREncoder(BaseModel):
    def __init__(self, num_layers, num_head, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.msa = nn.ModuleList()
        self.mlp = nn.ModuleList()
        for n in range(num_layers):
            self.msa.append(nn.Sequential(*[
                nn.LayerNorm(hidden_size),
                Attention(num_head=num_head, dim=hidden_size)
            ]))
            self.mlp.append(nn.Sequential(*[
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU()
                #nn.Linear(hidden_size, hidden_size)
            ]))
        
    
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.msa[i](x) + x
            x = self.mlp[i](x) + x
        return x
    

class ViTModel(BaseModel):
    def __init__(self, image_size=(3, 32, 32), patch_size=16, num_classes=1000, num_layers = 12, hidden_size=768, MLP_size=3072, Heads=12):
        super().__init__()
        
        c =  image_size[0]
        ph = patch_size
        pw = patch_size
        patch_num = image_size[1] * image_size[2] // patch_size**2
        
        self.img_to_patch = Rearrange('b c (h ph) (w pw) -> b (h w) (c ph pw)', c=c, ph=ph, pw=pw)
        self.linear_proj = nn.Linear(c*ph*pw, hidden_size)
        self.cls_token = nn.Parameter(torch.rand(1,1,hidden_size) ,requires_grad=True)
        self.position_embedding = nn.Parameter(torch.rand(1, patch_num+1, hidden_size), requires_grad=True)
        
        self.transformer_encoder =  TREncoder(num_layers, Heads, hidden_size)
        
        self.mlp_head = nn.Sequential(*[
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, MLP_size),
            nn.Linear(MLP_size, num_classes)
        ])
        
    def forward(self, img):
        x = self.img_to_patch(img)
        x = self.linear_proj(x)
        
        cls_token = einops.repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_token,x), dim=1)
        x += self.position_embedding
        
        x = self.transformer_encoder(x)
        
        x_cls = x[:, 0, :]
        y = self.mlp_head(x_cls)
        return y
        
        
    
    