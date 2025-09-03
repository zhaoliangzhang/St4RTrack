#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for DUSt3R
# --------------------------------------------------------
from dust3r.training import get_args_parser, train, load_model_with_hf_support
import croco.utils.misc as misc  # noqa
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.mode.startswith('eval'):
        misc.init_distributed_mode(args)
        global_rank = misc.get_rank()
        world_size = misc.get_world_size()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # fix the seed
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = not args.disable_cudnn_benchmark
        model, _ = load_model_with_hf_support(args, device)
        os.makedirs(args.output_dir, exist_ok=True)

        exit(0)
    train(args)