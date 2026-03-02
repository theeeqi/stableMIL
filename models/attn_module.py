import torch 
import torch.nn as nn
from d2d_rope import apply_rotary_emb
import numpy as np
import torch.nn.functional as F
import time
import warnings
import math

warnings.filterwarnings("ignore")
NEGATIVE_INFINITY = -int(1e9)

    
class flex_doucment_Attention(nn.Module):

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        k_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads 
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5


        self.kv = nn.Linear(dim, 2*dim, bias=k_bias) 
        self.query = nn.Parameter(torch.randn([1,self.head_dim*self.num_heads])) 
        nn.init.normal_(self.query,std=1e-6)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self,x,coords,region_indices):
       
        device = x.device
        B,N,C = x.shape
        
        unique_labels, inverse_indices = torch.unique(region_indices, return_inverse=True)
       
        region_indices = inverse_indices

        bin_count = torch.bincount(region_indices)
        bin_count = bin_count[bin_count > 0]
       
        q = self.query.unsqueeze(0).expand(B,N,-1)

        kv = self.kv(x).reshape(B,-1,2,self.num_heads,self.head_dim).permute(2,0,3,1,4)
      
        k,v=kv[0],kv[1]

        q, k = self.q_norm(q), self.k_norm(k)
        q = q.reshape(B,self.num_heads,-1,self.head_dim) 

        coords = coords.reshape(B,1,-1,2).expand([-1,self.num_heads,-1,-1])
        
        split_coords = torch.split(coords, bin_count.tolist(),dim=2)
        split_q = torch.split(q,bin_count.tolist(),dim=2)
        split_k = torch.split(k,bin_count.tolist(),dim=2)
        split_v = torch.split(v,bin_count.tolist(),dim=2)
        split_x1 = []
        split_coords1 = [] 
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, 
            enable_math=True,
            enable_mem_efficient=True,
            ):
            for i in range(len(bin_count)):    
                x_1 = F.scaled_dot_product_attention(split_q[i],split_k[i],split_v[i],scale=self.scale)
                coords_1 = F.scaled_dot_product_attention(split_q[i],split_k[i],split_coords[i],scale=self.scale)
                x_1 = x_1.permute(0,2,1,3).reshape(B,-1,C)
                x_1 = self.proj_drop(self.proj(x_1))    
                x_1 = torch.mean(x_1,dim=1,keepdim=True)

                coords_1 = torch.mean(coords_1,dim=2,keepdim=True)
                
                split_x1.append(x_1)
                split_coords1.append(coords_1)        
                

        x_1 = torch.concat(split_x1,dim=1)
        coords_1 = torch.concat(split_coords1,dim=2)

        return x_1,coords_1

    
class pile_attention(nn.Module):


    def __init__(self,
            dim: int = 768,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            k_neighbors = 8,
            max_dist = np.sqrt(2)
            ##周围n个邻居
            ) -> None:
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.max_dist = max_dist



class pile_attention(nn.Module):
    def __init__(self,
            dim: int = 768,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            k_neighbors = 8,
            max_dist = 6*np.sqrt(2)

            ) -> None:
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.max_dist = max_dist


    def forward(self,x,seman_tokens,coords,seman_coords,attn_mask=None,return_attn=False):

        assert attn_mask != None , "Attn mask should not be None for window attention"
        B, N_1, C= x.shape

        k_neighbors = min(N_1 - 1 ,self.k_neighbors)

        device = x.device

        x = torch.concat([x,seman_tokens],dim=1)
        coords = coords.unsqueeze(0).expand(-1,self.num_heads,-1,-1)

        coords = torch.concat([coords,seman_coords],dim=2)
    
        B, N, C = x.shape
        
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #[3,B,H,N,dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  


        q, k = apply_rotary_emb(q, k, coords=coords, use_random_project=self.training)


        q = self.q_norm(q)
        k = self.k_norm(k) 

        with torch.backends.cuda.sdp_kernel(
                enable_flash=True, 
                enable_math=True, 
                enable_mem_efficient=True
                ):
            out_fuse = F.scaled_dot_product_attention(
                q[:,:,N_1:,:], k, v,
                attn_mask=None,
                dropout_p = self.attn_drop_p, 
                is_causal = False, 
                scale = self.scale)
            out = out_fuse #N_K

            flag = attn_mask[1] 
            attn_mask = attn_mask[0]
            attn_mask = attn_mask.reshape(1,1,N_1,k_neighbors + 1,1).expand(-1,self.num_heads,-1,-1,C // self.num_heads) #[B,H,N,k,dim]
            pile_q = q[:,:,:N_1,:].unsqueeze(3)  # [batch_size, num_heads, N, 1, dim]

            pile_k = torch.gather(k[:,:,:N_1,:].unsqueeze(3).expand(-1,-1,-1,k_neighbors + 1, -1), 2,attn_mask)  # [batch_size, num_heads, N, k, dim]
            pile_v = torch.gather(v[:,:,:N_1,:].unsqueeze(3).expand(-1,-1,-1,k_neighbors + 1, -1), 2,attn_mask)
            pile_k = torch.concat([pile_k,k[:,:,N_1:,:].unsqueeze(2).expand(-1,-1,N_1,-1,-1)],dim=3)
            pile_v = torch.concat([pile_v,v[:,:,N_1:,:].unsqueeze(2).expand(-1,-1,N_1,-1,-1)],dim=3)

            # [N, K] -> [B, 1, N, K]
            local_mask = flag.unsqueeze(0).unsqueeze(1).expand(-1,self.num_heads,-1,-1)
            global_mask = torch.zeros([B, self.num_heads , pile_k.shape[2],pile_k.shape[3] - self.k_neighbors - 1])
            con_mask = torch.cat([local_mask,global_mask],dim=-1)
            full_mask = torch.zeros(pile_k.shape[:-1])
            full_mask = full_mask.masked_fill(con_mask == -1, float('-inf')).unsqueeze(-2)
            
            pile_out = F.scaled_dot_product_attention(
                pile_q, pile_k, pile_v,
                attn_mask=full_mask,
                dropout_p = self.attn_drop_p, 
                is_causal = False, 
                scale = self.scale)

            
            pile_out = pile_out.squeeze(dim=3)
            out = torch.concat([pile_out,out_fuse],dim=2)

            
        out = out.permute(0,2,1,3).reshape(B,-1,C)
        x = self.proj(out)
        x = self.proj_drop(x)


        return x
