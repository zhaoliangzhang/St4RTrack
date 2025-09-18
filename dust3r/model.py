# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
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

class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
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
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 arch_mode = 'VanillaDust3r',
                 rope_mode = 'full_3d', #full_3d, mix_3d
                 **croco_kwargs):
        
        croco_kwargs['arch_mode'] = arch_mode
        croco_kwargs['rope_mode'] = rope_mode
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.arch_mode = arch_mode
        self.random_init_lr_scale = random_init_lr_scale
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.head1_pretrained_path = head1_pretrained_path
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, head_type1, **croco_kwargs)
        self.set_freeze(freeze)
        self.set_random_init(random_init)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
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
                # Use Kaiming initialization for linear and conv layers
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                # Set scaled learning rate for randomly initialized parameters
                m.weight.lr_scale = self.random_init_lr_scale
                if m.bias is not None:
                    m.bias.lr_scale = self.random_init_lr_scale
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                # Initialize normalization layers
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                # Set scaled learning rate for randomly initialized parameters
                m.weight.lr_scale = self.random_init_lr_scale
                m.bias.lr_scale = self.random_init_lr_scale
            elif isinstance(m, nn.ConvTranspose2d):
                # set uniform initialization for conv transpose layers
                torch.nn.init.uniform_(m.weight, -0.02, 0.02)
                if m.bias is not None:
                    torch.nn.init.uniform_(m.bias, -0.02, 0.02)

                m.weight.lr_scale = self.random_init_lr_scale
                if m.bias is not None:
                    m.bias.lr_scale = self.random_init_lr_scale
            
        for module in to_be_random_init[random_init]:
            module.apply(init_weights)

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, head_type1, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type1 = head_type1
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
    
        self.downstream_head1 = head_factory(head_type1, output_mode, self, has_conf=bool(conf_mode))
        if self.head1_pretrained_path:
            self.downstream_head1.load_state_dict(torch.load(self.head1_pretrained_path, weights_only=False), strict=False)
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape, mask=None, seq_id=None):
        # embed the image into patches  (x has size B x Npatches x C)
    
        x, pos3d = self.patch_embed(image, true_shape=true_shape, seq_id=seq_id)

        pos = pos3d[:,:,1:] 
        B,N,C = x.size()
        if mask is not None:
            assert mask.shape[1] * mask.shape[2] == N
            mask = mask.view(B, -1)
            x = x[mask].view(B, -1, C)
            posvis = pos[mask].view(B, -1, 2)
        else:
            posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None, pos3d

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, mask1=None, mask2=None):
        if img1.shape[-2:] == img2.shape[-2:] and (mask1 is None) and (mask2 is None):
            out, pos, _, pos3d = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
            pos3d1, pos3d2 = pos3d.chunk(2, dim=0)
        else:
            out, pos, _, pos3d1 = self._encode_image(img1, true_shape1, mask=mask1)
            out2, pos2, _, pos3d2 = self._encode_image(img2, true_shape2, mask=mask2)
        return out, out2, pos, pos2, pos3d1, pos3d2

    def _encode_image_sequence(self, imgs, true_shapes, mask=None):
        if len(imgs) == 0:
            return [], [], []
            
        if all(img.shape[-2:] == imgs[0].shape[-2:] for img in imgs) and mask is None:
            # If all images have same shape and no masks, batch them together
            stacked_imgs = torch.cat(imgs, dim=0)
            stacked_shapes = torch.cat(true_shapes, dim=0)  
            out, pos, _, pos3d = self._encode_image(stacked_imgs, stacked_shapes)
            # Split back into individual tensors
            outs = list(out.chunk(len(imgs), dim=0))
            poses = list(pos.chunk(len(imgs), dim=0))
            pos3ds = list(pos3d.chunk(len(imgs), dim=0))
        else:
            # Encode each image separately if shapes differ or have masks
            outs = []
            poses = []
            for img, true_shape in zip(imgs, true_shapes):
                out, pos, _ = self._encode_image(img, true_shape, mask=mask)
                outs.append(out)
                poses.append(pos)
                
        return outs, poses, None

    def _encode_symmetrized(self, view1, view2):
    
        img1 = view1['img']
        img2 = view2['img']
        
        B = img1.shape[0]  
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        # Remove the extra dimension from shape tensors
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1)).squeeze(1)  # [B, 2]
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1)).squeeze(1)  # [B, 2]

        feat1, feat2, pos1, pos2, pos3d1, pos3d2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2), (pos3d1, pos3d2)

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

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2), (pos3d1, pos3d2) = self._encode_symmetrized(view1, view2)
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, pos3d1, pos3d2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
    
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
