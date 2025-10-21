# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class for Robot Jacobian Field input
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed
import torch.nn as nn
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa
from models.blocks import Attention as FastAttention
from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
import math

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def load_model(model_path, device, verbose=False):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)

    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereoJacobian(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d", "robot", "jacobian"],
):
    """DUSt3R model for robot Jacobian field prediction.
    
    This model takes:
    - Image from views1 (reference frame) - goes to both encoders
    
    The model processes the same image through both encoders. The first head 
    outputs a Jacobian field that maps joint actions to RGB changes.
    The second head outputs 3D points for consistency.
    """

    def __init__(self,
                 output_mode='jacobian',
                 head_type1='jacobian',
                 head1_pretrained_path=None,
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 random_init='none',
                 random_init_lr_scale=1.0,
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',
                 arch_mode='VanillaDust3r',
                 rope_mode='full_3d',
                 action_dim=8,  # Robot action dimension
                 **croco_kwargs):
        
        croco_kwargs['arch_mode'] = arch_mode
        croco_kwargs['rope_mode'] = rope_mode
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # Robot-specific initialization
        self.arch_mode = arch_mode
        self.random_init_lr_scale = random_init_lr_scale
        self.action_dim = action_dim
        
        # Create second decoder (for action processing)
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.head1_pretrained_path = head1_pretrained_path
        
        # Set up heads
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, head_type1, **croco_kwargs)
        self.set_freeze(freeze)
        self.set_random_init(random_init)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereoJacobian, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # Handle loading from standard DUSt3R checkpoints
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
            'encoder_and_head2': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks2, self.downstream_head2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def set_random_init(self, random_init):
        self.random_init = random_init
        to_be_random_init = {
            'none': [],
            'head1': [self.downstream_head1],
        }
        
        def init_weights(m):
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                m.weight.lr_scale = self.random_init_lr_scale
                if m.bias is not None:
                    m.bias.lr_scale = self.random_init_lr_scale
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                m.weight.lr_scale = self.random_init_lr_scale
                m.bias.lr_scale = self.random_init_lr_scale
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.uniform_(m.weight, -0.02, 0.02)
                if m.bias is not None:
                    torch.nn.init.uniform_(m.bias, -0.02, 0.02)
                m.weight.lr_scale = self.random_init_lr_scale
                if m.bias is not None:
                    m.bias.lr_scale = self.random_init_lr_scale
            
        for module in to_be_random_init[random_init]:
            module.apply(init_weights)

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, head_type1, patch_size, img_size, **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type1 = head_type1
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        
        # Allocate heads - pass joint_dim for Jacobian head
        self.downstream_head1 = head_factory(head_type1, output_mode, self, has_conf=bool(conf_mode), joint_dim=self.action_dim)
        if self.head1_pretrained_path:
            self.downstream_head1.load_state_dict(torch.load(self.head1_pretrained_path, weights_only=False), strict=False)
        self.downstream_head2 = head_factory(head_type, 'pts3d', self, has_conf=bool(conf_mode))
        
        # Magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape, mask=None, seq_id=None):
        """Encode image using standard patch embedding"""
        x, pos3d = self.patch_embed(image, true_shape=true_shape, seq_id=seq_id)
        pos = pos3d[:,:,1:] 
        B, N, C = x.size()
        if mask is not None:
            assert mask.shape[1] * mask.shape[2] == N
            mask = mask.view(B, -1)
            x = x[mask].view(B, -1, C)
            posvis = pos[mask].view(B, -1, 2)
        else:
            posvis = pos
        
        # Apply transformer encoder
        for blk in self.enc_blocks:
            x = blk(x, posvis)
        x = self.enc_norm(x)
        return x, pos, None, pos3d


    def _encode_jacobian_inputs(self, view1, view2):
        """Encode inputs for Jacobian model: image goes to both encoders"""
        # Extract images from view1 (contains [img0, img1])
        img = view1['img']
        
        # Check if we have [B, S, C, H, W] format and reshape to [B*S, C, H, W]
        if img.ndim == 5:  # [B, S, C, H, W]
            B, S = img.shape[:2]
            img = img.view(B * S, *img.shape[2:])  # [B*S, C, H, W]
            
            # Adjust true_shape accordingly
            if 'true_shape' in view1:
                true_shape = view1['true_shape']  # [B, S, 2] or [B, 2]
                if true_shape.ndim == 3:  # [B, S, 2]
                    true_shape = true_shape.view(B * S, 2)  # [B*S, 2]
                else:  # [B, 2]
                    true_shape = true_shape.repeat_interleave(S, dim=0)  # [B*S, 2]
                shape1 = true_shape
            else:
                shape1 = torch.tensor(img.shape[-2:])[None].repeat(B * S, 1)  # [B*S, 2]
        else:  # Standard [B, C, H, W] format
            batch_size = img.shape[0]
            # Get true shapes
            shape1 = view1.get('true_shape', torch.tensor(img.shape[-2:])[None].repeat(batch_size, 1))
            if shape1.ndim == 2 and shape1.shape[0] == 1:
                shape1 = shape1.squeeze(0).repeat(batch_size, 1)
        
        # Encode images through first encoder (for Jacobian field prediction)
        img_feat1, img_pos1, _, img_pos3d1 = self._encode_image(img, shape1)
        
        # Encode images through second encoder (for 3D point prediction)
        img_feat2, img_pos2, _, img_pos3d2 = self._encode_image(img, shape1)
        
        # For Jacobian model, we use image features for both encoders
        feat1 = img_feat1   # Image features for Jacobian head
        feat2 = img_feat2   # Image features for 3D point head
        pos1 = img_pos1     # Image positions
        pos2 = img_pos2     # Image positions
        pos3d1 = img_pos3d1 # Image 3D positions
        pos3d2 = img_pos3d2 # Image 3D positions

        # Return in the same format as _encode_image_pairs
        return (shape1, shape1), (feat1, feat2), (pos1, pos2), (pos3d1, pos3d2)

    def _downstream_head(self, head_num, decout, img_shape):
        """Apply downstream head"""
        B, S, D = decout[-1].shape
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def _decoder_pre(self, f1, pos1, f2, pos2, pos3d1, pos3d2):
        final_output = [(f1, f2)]  # before projection
    
        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)
        return f1, f2, pos1, pos2, pos3d1, pos3d2, final_output[0]

    def _decoder_post(self, f1, f2, pos2d1, pos2d2, pos3d1, pos3d2, original_output):
    
        final_output = [original_output, (f1, f2)]
        if self.arch_mode != 'TempDust3r' or self.rope_mode == 'mix_3d':
            pos1 = pos2d1
            pos2 = pos2d2
        else:
            pos1 = pos3d1
            pos2 = pos3d2
        
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]
        
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _decoder(self, f1, pos1, f2, pos2, pos3d1, pos3d2):
    
        f1, f2, pos1, pos2, pos3d1, pos3d2, original_output = self._decoder_pre(f1, pos1, f2, pos2, pos3d1, pos3d2)
        return self._decoder_post(f1, f2, pos1, pos2, pos3d1, pos3d2, original_output)

    def forward(self, view1, view2):
        """Forward pass for robot Jacobian field prediction"""
        # Encode inputs: image goes to both encoders, action goes to one encoder
        (shape1, shape2), (feat1, feat2), (pos1, pos2), (pos3d1, pos3d2) = self._encode_jacobian_inputs(view1, view2)
        
        # Decode using standard decoder
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, pos3d1, pos3d2)

        # Apply heads
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)  # Jacobian field
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)  # 3D points
    
        # Rename output for consistency
        res2['pts3d_in_other_view'] = res2.pop('pts3d')
        return res1, res2
