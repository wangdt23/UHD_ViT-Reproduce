# src/models/vit.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

class PatchEmbed(nn.Module):
    """patch embedding: cut image into patches and embed to feature space"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # [B, C, H, W]
        return x

class Downsample(nn.Module):
    """downsample: reduce spatial size and increase feature dimension"""
    def __init__(self, in_embed, out_embed):
        super().__init__()
        self.reduction = nn.Conv2d(in_embed, out_embed, kernel_size=2, stride=2)

    def forward(self, x):
        return self.reduction(x)

class HierarchicalViT(nn.Module):
    '''Hierarchical Vision Transformer with patch embedding, multiple stages of transformer blocks and downsampling'''
    def __init__(self, in_chans=3, num_classes=10, embed_dims=[64, 128, 256], depths=[2, 2, 2]):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=4, in_chans=in_chans, embed_dim=embed_dims[0])
        
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            # add transformer blocks for this stage
            stage = nn.Sequential(*[
                Block(dim=embed_dims[i], num_heads=8) for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            # add downsample layer between stages (except after the last stage)
            if i < len(depths) - 1:
                self.stages.append(Downsample(embed_dims[i], embed_dims[i+1]))

        self.head = nn.Linear(embed_dims[-1], num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")

        x = self.patch_embed(x)
        # print(f"After patch embedding: {x.shape}")

        for i,stage in enumerate(self.stages):
            if isinstance(stage, nn.Sequential): # Transformer blocks
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2) # [batch_size, num_patches, embed_dim] for transformer
                # print(f"{i}Stage{i}Input from Transformer blocks: {x.shape}")

                x = stage(x)
                x = x.transpose(1, 2).reshape(B, C, H, W) # reshape back to [B, C, H, W]
                # print(f"{i}Stage{i}Output from Transformer blocks: {x.shape}")
            else: # Downsample layer
                x = stage(x)
                # print(f"{i}After downsampling: {x.shape}")
        
        x = self.avgpool(x).flatten(1)
        # print(f"After global average pooling: {x.shape}")
        x =  self.head(x)
        # print(f"Output shape: {x.shape}") 
        # import sys; sys.exit() # for debugging, remove this line after confirming output shape is correct
        return x