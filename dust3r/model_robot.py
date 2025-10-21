# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class for Robot Action-Image input
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

class ActionEmbedding(nn.Module):
    """Action embedding module to convert robot actions and camera pose to patch-like features"""
    
    def __init__(self, action_dim, embed_dim, patch_size=16, img_size=(512, 288)):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Calculate number of patches in the image
        # img_size is (width, height), so img_size[0] = width, img_size[1] = height
        self.num_patches_h = img_size[1] // patch_size  # height // patch_size
        self.num_patches_w = img_size[0] // patch_size  # width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Action to embedding projection
        self.action_proj = nn.Linear(action_dim, embed_dim)
        
        # Camera pose to embedding projection (4x4 matrix flattened to 16D)
        self.camera_pose_proj = nn.Linear(16, embed_dim)
        
        # Combined projection for action + camera pose
        self.combined_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # Learnable spatial modulation for actions
        # Instead of broadcasting the same action to all patches, 
        # we create spatial variants of the action
        self.spatial_modulation = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # Initialize weights
        nn.init.trunc_normal_(self.spatial_modulation, std=0.02)
        nn.init.xavier_uniform_(self.action_proj.weight)
        nn.init.xavier_uniform_(self.camera_pose_proj.weight)
        nn.init.xavier_uniform_(self.combined_proj.weight)
        nn.init.constant_(self.action_proj.bias, 0)
        nn.init.constant_(self.camera_pose_proj.bias, 0)
        nn.init.constant_(self.combined_proj.bias, 0)
    
    def forward(self, actions, camera_pose=None):
        """
        Args:
            actions: [B, action_dim] - Robot actions
            camera_pose: [B, 4, 4] - Camera pose matrices (optional)
        Returns:
            action_features: [B, num_patches, embed_dim] - Action features
            action_pos: [B, num_patches, 2] - 2D positions for action patches
        """
        B = actions.shape[0]
        
        # Project actions to embedding dimension
        action_embed = self.action_proj(actions)  # [B, embed_dim]
        
        # Project camera pose to embedding dimension if provided
        if camera_pose is not None:
            # Flatten camera pose matrices from [B, 4, 4] to [B, 16]
            camera_pose_flat = camera_pose.view(B, -1).contiguous()  # [B, 16]
            camera_pose_embed = self.camera_pose_proj(camera_pose_flat)  # [B, embed_dim]
            
            # Combine action and camera pose embeddings
            combined_embed = torch.cat([action_embed, camera_pose_embed], dim=-1).contiguous()  # [B, embed_dim * 2]
            combined_embed = self.combined_proj(combined_embed)  # [B, embed_dim]
        else:
            # Use only action embedding if no camera pose provided
            combined_embed = action_embed
        
        # Create spatially modulated action features
        # Each patch gets a slightly different version of the combined action+pose
        action_features = combined_embed.unsqueeze(1).expand(B, self.num_patches, self.embed_dim)  # [B, num_patches, embed_dim]
        action_features = action_features + self.spatial_modulation  # Add learnable spatial modulation
        
        # Generate 2D positions for action patches
        # Create grid positions for patches
        y_coords = torch.arange(self.num_patches_h, device=actions.device)
        x_coords = torch.arange(self.num_patches_w, device=actions.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten and stack coordinates
        action_pos = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1).contiguous()  # [num_patches, 2]
        action_pos = action_pos.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, num_patches, 2]
        
        return action_features, action_pos

class AsymmetricCroCo3DStereoRobot(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d", "robot"],
):
    """DUSt3R model for robot action-image input.
    
    Instead of two images, this model takes:
    - Image from views1 (reference frame)
    - Action from views2 (robot joint positions difference)
    
    The model processes the image through the standard encoder and
    the action through a special action embedding module.
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type1='dpt',
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
        
        # Create action embedding module
        self.action_embed = ActionEmbedding(
            action_dim=action_dim,
            embed_dim=self.enc_embed_dim,
            patch_size=croco_kwargs.get('patch_size', 16),
            img_size=croco_kwargs.get('img_size', (512, 288))
        )
        
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
            return super(AsymmetricCroCo3DStereoRobot, cls).from_pretrained(pretrained_model_name_or_path, **kw)

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
        
        # Allocate heads
        self.downstream_head1 = head_factory(head_type1, output_mode, self, has_conf=bool(conf_mode))
        if self.head1_pretrained_path:
            self.downstream_head1.load_state_dict(torch.load(self.head1_pretrained_path, weights_only=False), strict=False)
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        
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
            mask = mask.view(B, -1).contiguous()
            x = x[mask].view(B, -1, C).contiguous()
            posvis = pos[mask].view(B, -1, 2).contiguous()
        else:
            posvis = pos
        
        # Apply transformer encoder
        for blk in self.enc_blocks:
            x = blk(x, posvis)
        x = self.enc_norm(x)
        return x, pos, None, pos3d

    def _encode_action(self, actions, camera_pose=None):
        """Encode robot actions and camera pose using action embedding"""
        # Get action features and positions (now includes camera pose)
        action_features, action_pos = self.action_embed(actions, camera_pose)  # [B, num_patches, embed_dim], [B, num_patches, 2]
        
        # Apply transformer encoder to action features
        for blk in self.enc_blocks:
            action_features = blk(action_features, action_pos)
        action_features = self.enc_norm(action_features)
        
        # Return same format as _encode_image: (features, pos, None, pos3d)
        # For actions, we don't have 3D positions, so use 2D positions as pos3d
        action_pos3d = torch.cat([action_pos, torch.zeros_like(action_pos[:, :, :1])], dim=-1).contiguous()  # [B, num_patches, 3]
        return action_features, action_pos, None, action_pos3d

    def _encode_robot_inputs(self, view1, view2):
        """Encode robot inputs: encode images and actions separately, then combine"""
        # Extract images from view1 (contains [img0, img1])
        img = view1['img']
        
        # Check if we have [B, S, C, H, W] format and reshape to [B*S, C, H, W]
        if img.ndim == 5:  # [B, S, C, H, W]
            B, S = img.shape[:2]
            img = img.view(B * S, *img.shape[2:]).contiguous()  # [B*S, C, H, W]]
            
            # Adjust true_shape accordingly
            if 'true_shape' in view1:
                true_shape = view1['true_shape']  # [B, S, 2] or [B, 2]
                if true_shape.ndim == 3:  # [B, S, 2]
                    true_shape = true_shape.view(B * S, 2).contiguous()  # [B*S, 2]
                else:  # [B, 2]
                    true_shape = true_shape.repeat_interleave(S, dim=0).contiguous()  # [B*S, 2]
                shape1 = true_shape
            else:
                shape1 = torch.tensor(img.shape[-2:])[None].repeat(B * S, 1)  # [B*S, 2]
        else:  # Standard [B, C, H, W] format
            batch_size = img.shape[0]
            # Get true shapes
            shape1 = view1.get('true_shape', torch.tensor(img.shape[-2:])[None].repeat(batch_size, 1))
            if shape1.ndim == 2 and shape1.shape[0] == 1:
                shape1 = shape1.squeeze(0).repeat(batch_size, 1)
        
        # Compute actions as view2 - view1 (joint position differences)
        joint_pos_view1 = view1['joint_pos']  # joint positions from view1
        joint_pos_view2 = view2['joint_pos']  # joint positions from view2
        
        # Extract camera pose from view2 (camera pose for the target view)
        camera_pose = view2.get('camera_pose', None)  # [B, 4, 4] or [B, S, 4, 4]
        
        # Handle action reshaping based on original image dimensions
        original_img = view1['img']
        if original_img.ndim == 5:  # We reshaped the image, so we need to reshape actions too
            # Actions: view2 - view1 (element-wise difference)
            actions = joint_pos_view2 - joint_pos_view1  # [batch_size, S, action_dim]
            # Use the same B and S variables from image reshaping
            actions = actions.view(B * S, -1).contiguous()  # [B*S, action_dim]
            
            # Handle camera pose reshaping
            if camera_pose is not None:
                if camera_pose.ndim == 4:  # [B, S, 4, 4]
                    camera_pose = camera_pose.view(B * S, 4, 4).contiguous()  # [B*S, 4, 4]
                elif camera_pose.ndim == 3:  # [B, 4, 4]
                    camera_pose = camera_pose.repeat_interleave(S, dim=0).contiguous()  # [B*S, 4, 4]
        else:
            # Actions: view2 - view1 (element-wise difference)
            actions = joint_pos_view2 - joint_pos_view1  # [batch_size, S, action_dim] or [batch_size, action_dim]
            if actions.ndim == 3:  # [batch_size, S, action_dim]
                actions = actions.view(actions.shape[0] * actions.shape[1], -1).contiguous()  # [batch_size*S, action_dim]
            
            # Handle camera pose reshaping for non-5D case
            if camera_pose is not None and camera_pose.ndim == 3:  # [B, 4, 4]
                if actions.ndim == 2:  # [batch_size, action_dim] - no sequence dimension
                    pass  # camera_pose is already correct shape [B, 4, 4]
                else:  # actions was reshaped from [batch_size, S, action_dim] to [batch_size*S, action_dim]
                    # camera_pose needs to be reshaped from [B, 4, 4] to [B*S, 4, 4]
                    batch_size = actions.shape[0] // (actions.shape[0] // camera_pose.shape[0])
                    S = actions.shape[0] // camera_pose.shape[0]
                    camera_pose = camera_pose.repeat_interleave(S, dim=0).contiguous()  # [B*S, 4, 4]
        
        # Encode images separately (same as model.py _encode_image_pairs)
        # Use the same batching optimization as model.py (lines 211-215)
        out, pos, _, pos3d = self._encode_image(img, shape1)
        
        # Encode actions and camera pose together - now actions is already reshaped to [B*S, action_dim]
        action_feat, action_pos, _, action_pos3d = self._encode_action(actions, camera_pose)  # [B*S, num_patches, embed_dim]
        # For robot model, we use action features as feat1 and image features as feat2
        feat1 = action_feat  # Action features
        feat2 = out          # Image features
        pos1 = action_pos    # Action positions
        pos2 = pos           # Image positions
        pos3d1 = action_pos3d  # Action 3D positions
        pos3d2 = pos3d       # Image 3D positions

        # print('feat1', feat1.shape)
        # print('feat2', feat2.shape)
        # print('pos1', pos1.shape)
        # print('pos2', pos2.shape)
        # print('pos3d1', pos3d1.shape)
        # print('pos3d2', pos3d2.shape)
        
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
        """Forward pass for robot action-image input"""
        # Encode robot inputs: encode separately then combine to match _encode_image_pairs format
        (shape1, shape2), (feat1, feat2), (pos1, pos2), (pos3d1, pos3d2) = self._encode_robot_inputs(view1, view2)
        
        # Decode using standard decoder (since shapes now match _encode_image_pairs)
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, pos3d1, pos3d2)

        # Apply heads
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
            # print('res1', res1['pts3d'].shape)
            # print('res2', res2['pts3d'].shape)
    
        # Rename output for consistency
        res2['pts3d_in_other_view'] = res2.pop('pts3d')
        return res1, res2
