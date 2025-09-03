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

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
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
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 flow_cond = False,
                 normalize_flow = False,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)
        flow_cond = False
        self.flow_cond = flow_cond
        self.normalize_flow = normalize_flow
        if self.flow_cond:
            print('Using RAFT for flow conditioning')
            self.flow_embed = ManyAR_PatchEmbed(img_size=croco_kwargs['img_size'], patch_size=croco_kwargs['patch_size'], in_chans=2, embed_dim=768, init='zero')
            self.flow_embed._init_weights()
            torch.nn.init.zeros_(self.flow_embed.proj.weight)
            torch.nn.init.zeros_(self.flow_embed.proj.bias)
            # self.flow_net = load_RAFT(model_path="third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth").to('cuda').eval()
            # don't update the weights of the flow_net
            # for param in self.flow_net.parameters():
            #     param.requires_grad = False

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
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape, mask=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        if mask is not None:
            # mask [1, H//16, W//16]
            assert mask.shape[1] * mask.shape[2] == N
            mask = mask.view(B, -1)
            x = x[mask].view(B, -1, C)
            posvis = pos[mask].view(B, -1, 2)
        else:
            posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, mask1=None, mask2=None):
        if img1.shape[-2:] == img2.shape[-2:] and (mask1 is None) and (mask2 is None):
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1, mask=mask1)
            out2, pos2, _ = self._encode_image(img2, true_shape2, mask=mask2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2, do_mask=False):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        if self.flow_cond:
            img1_flow = (img1 + 1) / 2
            img2_flow = (img2 + 1) / 2
            flow1 = self.flow_net(img1_flow*255, img2_flow*255, iters=20, test_mode=True)[1]
            flow2 = self.flow_net(img2_flow*255, img1_flow*255, iters=20, test_mode=True)[1]
            if self.normalize_flow:
                # normalize flow to [-1, 1]
                flow1 = flow1 / (shape1.flip(dims=[-1]).unsqueeze(-1).unsqueeze(-1) / 2)
                flow2 = flow2 / (shape2.flip(dims=[-1]).unsqueeze(-1).unsqueeze(-1) / 2)
            flow1, flow_pos = self.flow_embed(flow1, true_shape=shape1)
            flow2, flow_pos = self.flow_embed(flow2, true_shape=shape2)
        else:
            flow1 = flow2 = None

        # warning! maybe the images have different portrait/landscape orientations
        if do_mask:
            mask1 = view1['patch_mask'] # [1, H//16, W//16]
            mask2 = view2['patch_mask']
        else:
            mask1 = mask2 = None
        if is_symmetrized(view1, view2) and not do_mask:
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            if not do_mask:
                feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)
            else:
                feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2, mask1, mask2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2), (mask1, mask2), (flow1, flow2)

    def _decoder(self, f1, pos1, f2, pos2, mask1=None, mask2=None, flow1=None, flow2=None, use_flow=None):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        if self.flow_cond and flow1 is not None and flow2 is not None:
            f1 = f1 + flow1 * use_flow.unsqueeze(-1).unsqueeze(-1) if use_flow is not None else f1 + flow1
            f2 = f2 + flow2 * use_flow.unsqueeze(-1).unsqueeze(-1) if use_flow is not None else f2 + flow2

        # add mask here
        if mask1 is not None:
            B, Nenc, D = f1.shape
            mask1 = mask1.view(B, -1).to(device=f1.device)
            Ntotal = mask1.shape[1]
            f1_ = self.mask_token.repeat(f1.shape[0], Ntotal, 1).to(dtype=f1.dtype, device=f1.device)
            f1_[mask1] = f1.view(B*Nenc, D)
            f1 = f1_.view(B, Ntotal, D)

            zero_token = torch.zeros(1, 1, original_D)
            original_f1 = zero_token.repeat(f1.shape[0], Ntotal, 1).to(dtype=f1.dtype, device=f1.device)
            original_f1[mask1] = final_output[0][0].view(B*Nenc, original_D)
            
        if mask2 is not None:
            B, Nenc, D = f2.shape
            mask2 = mask2.view(B, -1).to(device=f2.device)
            Ntotal = mask2.shape[1]
            f2_ = self.mask_token.repeat(f2.shape[0], Ntotal, 1).to(dtype=f2.dtype, device=f2.device)
            f2_[mask2] = f2.view(B*Nenc, D)
            f2 = f2_.view(B, Ntotal, D)

            original_f2 = zero_token.repeat(f1.shape[0], Ntotal, 1).to(dtype=f2.dtype, device=f2.device)
            original_f2[mask2] = final_output[0][1].view(B*Nenc, original_D)
            
            final_output.append((original_f1, original_f2))
            del final_output[0]

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2, do_mask=False):
        # encode the two images --> B,S,D
        prob_use_flow = 0.0 # TODO: for testing
        use_flow = torch.rand(view1['img'].shape[0]) < prob_use_flow if self.training else torch.ones(view1['img'].shape[0]).bool()
        use_flow = use_flow.to(view1['img'].device)
        (shape1, shape2), (feat1, feat2), (pos1, pos2), (mask1, mask2), (flow1, flow2) = self._encode_symmetrized(view1, view2, do_mask=do_mask)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, mask1, mask2, flow1, flow2, use_flow)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
