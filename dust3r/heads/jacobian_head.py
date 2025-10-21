# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Jacobian head implementation for DUST3R Robot Model
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F


class JacobianHead(nn.Module):
    """ 
    Jacobian head for robot model that outputs Jacobian field
    Each token outputs: - 16x16 Jacobian field (3 RGB channels x joint_dim)
    """
    
    def __init__(self, net, joint_dim=8, has_conf=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.joint_dim = joint_dim
        self.has_conf = has_conf
        
        # Output 3 RGB channels * joint_dim for Jacobian field
        # If has_conf, also output confidence
        output_channels = 3 * joint_dim + (1 if has_conf else 0)
        self.proj = nn.Linear(net.dec_embed_dim, output_channels * self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # Extract Jacobian field
        feat = self.proj(tokens)  # [B, S, output_channels * patch_size^2]
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # [B, 3*joint_dim, H, W]
        
        # Reshape to [B, H, W, 3, joint_dim] format
        feat = feat.permute(0, 2, 3, 1)  # [B, H, W, 3*joint_dim]
        feat = feat.view(B, H, W, 3, self.joint_dim)  # [B, H, W, 3, joint_dim]
        
        result = {'jacobian_field': feat}
        
        # If confidence is enabled, extract it from the last channel
        if self.has_conf:
            # For confidence, we need to extract from the original feat before reshaping
            feat_conf = self.proj(tokens)  # [B, S, output_channels * patch_size^2]
            feat_conf = feat_conf.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
            feat_conf = F.pixel_shuffle(feat_conf, self.patch_size)  # [B, output_channels, H, W]
            conf = feat_conf[:, 3*self.joint_dim:3*self.joint_dim+1, :, :]  # [B, 1, H, W]
            conf = conf.squeeze(1)  # [B, H, W]
            result['conf'] = conf
            
        return result
