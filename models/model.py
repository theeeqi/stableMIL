import sys 
import torch
from timm.models.layers import DropPath,Mlp
from timm.models.vision_transformer import LayerScale
import torch.nn as nn 
from attn_module import pile_attention,flex_doucment_Attention
import numpy as np
import torch.nn.functional as F
import torch._dynamo


torch._dynamo.config.suppress_errors = True

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class attn_block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            use_ffn = True,
            mlp_layer=Mlp,
            k_neighbors = 8
    ) -> None:
        ...
        super().__init__()
        
        self.use_ffn = use_ffn 
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)

        self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.parti_attention = pile_attention(dim,num_heads,qkv_bias,qk_norm,attn_drop,proj_drop,norm_layer,k_neighbors=k_neighbors)

    def forward(self,x,seman_x,coords,seman_coords,mask,return_att=False):
        
        B,N,C = x.shape
        
        if return_att:
            out,x = self.parti_attention(x,seman_x,coords,seman_coords,mask,return_att)
            return out,x
        
        x1 = torch.concat([x,seman_x],dim=1)
        x = self.parti_attention(x,seman_x,coords,seman_coords,mask)
        x = x1 + self.drop_path1(self.ls1(x))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        return x[:,:N,:],x[:,N:,:],coords,seman_coords,mask
    
class sequentialV2(nn.Sequential):
    def __init__(self,*args):
        super(sequentialV2,self).__init__(*args)
    def forward(self,x,seman_x,coords,seman_coords,mask):
        for module in self:
            x,seman_x,coords,seman_coords,mask= module(x,seman_x,coords,seman_coords,mask)
        return torch.concat([x,seman_x],dim=1)
##
class stableMIL(nn.Module):
    def __init__(
        self,
        depth :int = 3,
        dim: int = 768,
        hidden_dim : int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        init_values: float = None,
        pre_norm: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        drop_path_rate: float = 0.,
        drop_rate: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        aggregate_num :int =  128,
        n_classes: int = 2,
        use_ffn : bool = True,
        use_fc_norm =  True,
        act_layer = nn.GELU,
        mlp_layer = Mlp,
        task = 'subtype',
        k_neighbors = 8,
        max_dist = 6*np.sqrt(2),
        ratio = 2,
        learnable_mapping = False,
    ) -> None:
        ...
        super().__init__()
        self.task = task
        self.max_dist = max_dist + 1e-4
        self.ratio = ratio
        self.aggregate_num = aggregate_num
        self.k_neighbors = k_neighbors
        self.num_heads   = num_heads
        self.norm_pre = norm_layer(dim) if pre_norm else nn.Identity()
        if learnable_mapping:
            self.maping = nn.Linear(ratio**2*dim,hidden_dim)
        else:
            self.maping = nn.Linear(dim,hidden_dim)
        self.learnable_mapping = learnable_mapping
        self.act1 = nn.GELU()

        self.ag_attn = flex_doucment_Attention(dim=hidden_dim,num_heads=num_heads,k_bias=qkv_bias,qk_norm=qk_norm,proj_drop=proj_drop,attn_drop=attn_drop,norm_layer=norm_layer)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = sequentialV2(*[
            attn_block(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                use_ffn = use_ffn,
                k_neighbors = k_neighbors
            )
            for i in range(depth)])

        
        self.norm = norm_layer(hidden_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(hidden_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(hidden_dim, n_classes) if n_classes > 0 else nn.Identity()

        self.apply(initialize_weights)

    def relocate(self,device = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.norm_pre.to(device)
        self.ag_attn.to(device)
        self.fuse.to(device)
        self.act1.to(device)
        self.blocks.to(device)  
        self.norm.to(device)
        self.fc_norm.to(device)

        self.head_drop.to(device)
        self.head.to(device)
    
    def fuse_token(self,x,coords,labels,sorted_idx):
        B,N,C = x.shape
        device = x.device

        x = x[0]
        
        x = x[sorted_idx]
        
        bin_count = torch.bincount(labels)

        bin_count = bin_count[bin_count>0]

        split_features = torch.split(x,bin_count.tolist())

        max_points = int(self.ratio**2)
        padded_feats =[]
        for i, feat in enumerate(split_features):
            num_points = feat.size(0)
            if num_points >= max_points:

                padded_feat = feat[:max_points]
            else:


                pad_value_f = feat.mean(dim=0).expand(max_points-num_points,-1)
                padded_feat = torch.concat([feat,pad_value_f],dim=0)

            padded_feats.append(padded_feat)


        padded_feats = torch.stack(padded_feats)
        N_1 = padded_feats.shape[0]
        
        if self.learnable_mapping:
            padded_feats = padded_feats.reshape(N_1,-1)
            x = self.act1(self.maping(padded_feats)).reshape(B,N_1,-1)
        
        else:
            padded_feats = padded_feats.permute(0,2,1).mean(dim=-1,keepdim=False)
            x = self.act1(self.maping(padded_feats)).reshape(B,N_1,-1)

        return x ,coords
    
    def forward_features(self,x,coords,fuse_labels,fuse_sorted_idx,region_indices,region_sorted_index,attention_mask_1,attention_mask_2,return_att=False):

        x,coords = self.fuse_token(x,coords,fuse_labels,fuse_sorted_idx)
        x = self.norm_pre(x)
        
        device = x.device
        x = x[:,region_sorted_index]
        coords = coords[:,region_sorted_index]
        region_labels = region_indices[region_sorted_index]
        attention_mask = (attention_mask_1,attention_mask_2)
        seman_x,seman_coords = self.ag_attn(x,coords,region_labels)
        
        x = self.blocks(x,seman_x,coords,seman_coords,attention_mask)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head_drop(x)


        return x
    
    def forward_subtyping(self,x): 
        logits = self.head(x)
        prob = F.softmax(logits, dim=-1)
        Y_hat = torch.topk(prob, 1, dim = 1)[1]
        return logits, prob, Y_hat

    def forward_prognosis(self,x):
        
        logits  = self.head(x)  
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat,

    def forward(self,x,coords,fuse_labels,fuse_sorted_idx,region_indices,region_sorted_index,attention_mask_1,attention_mask_2,return_att=False):

        x = self.forward_features(x,coords,fuse_labels,fuse_sorted_idx,region_indices,region_sorted_index,attention_mask_1,attention_mask_2)
        if self.task == 'subtype':
            return self.forward_subtyping(x)
        elif self.task == 'survival':
            return self.forward_prognosis(x)
        else:
            raise ValueError('Task not supported')

