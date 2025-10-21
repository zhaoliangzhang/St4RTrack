#!/usr/bin/env python3
"""
Example usage of the Robot Action-Image DUSt3R model
"""
import torch
import numpy as np
from dust3r.model_robot import AsymmetricCroCo3DStereoRobot

def create_sample_robot_data(batch_size=2, img_size=(512, 288), action_dim=7):
    """Create sample robot data for testing with DataLoader batching"""
    
    # Create sample views1 (images + joint positions)
    # After DataLoader batching: [batch_size, B=2, 3, H, W]
    views1 = {
        'img': torch.randn(batch_size, 2, 3, img_size[1], img_size[0]),  # [batch_size, B=2, 3, H, W]
        'joint_pos': torch.randn(batch_size, 2, action_dim),  # [batch_size, B=2, action_dim] - joint positions
        'true_shape': torch.tensor([[img_size[1], img_size[0]]] * batch_size),  # [batch_size, 2]
        'dataset': 'Robot',
        'supervised_label': torch.ones(batch_size),
    }
    
    # Create sample views2 (joint positions only)
    # After DataLoader batching: [batch_size, B=2, action_dim]
    views2 = {
        'joint_pos': torch.randn(batch_size, 2, action_dim),  # [batch_size, B=2, action_dim] - joint positions
        'dataset': 'Robot',
        'supervised_label': torch.ones(batch_size),
    }
    
    return views1, views2

def main():
    print("Robot Action-Image DUSt3R Model Example")
    print("=" * 50)
    
    # Create model
    model = AsymmetricCroCo3DStereoRobot(
        output_mode='pts3d',
        head_type1='dpt',
        head_type='linear',
        depth_mode=('exp', -float('inf'), float('inf')),
        conf_mode=('exp', 1, float('inf')),
        freeze='none',
        landscape_only=True,
        patch_embed_cls='PatchEmbedDust3R',
        arch_mode='VanillaDust3r',
        rope_mode='full_3d',
        action_dim=7,  # Robot action dimension
        img_size=(512, 288),
        patch_size=16,
        enc_embed_dim=768,
        enc_depth=12,
        enc_num_heads=12,
        dec_embed_dim=512,
        dec_depth=8,
        dec_num_heads=16,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create sample data
    batch_size = 2
    views1, views2 = create_sample_robot_data(batch_size=batch_size, action_dim=7)
    
    print(f"Input shapes (after DataLoader batching):")
    print(f"  views1['img']: {views1['img'].shape} (batch_size={batch_size}, B=2, contains [img0, img1])")
    print(f"  views1['joint_pos']: {views1['joint_pos'].shape} (batch_size={batch_size}, B=2, joint positions)")
    print(f"  views2['joint_pos']: {views2['joint_pos'].shape} (batch_size={batch_size}, B=2, joint positions)")
    print(f"  Actions computed as: views2['joint_pos'] - views1['joint_pos']")
    print(f"  Image encoding: torch.cat((img1, img2), dim=0) then chunk(2) - same as model.py lines 211-215")
    print(f"  Decoder batching: [img1, action1, img2, action2] as batch with cross-attention")
    print(f"  Cross-attention in decoder: img1 <-> action1, img2 <-> action2")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            res1, res2 = model(views1, views2)
            
            print(f"\nOutput shapes:")
            print(f"  res1 keys: {list(res1.keys())}")
            print(f"  res2 keys: {list(res2.keys())}")
            
            for key, value in res1.items():
                if isinstance(value, torch.Tensor):
                    print(f"  res1['{key}']: {value.shape}")
            
            for key, value in res2.items():
                if isinstance(value, torch.Tensor):
                    print(f"  res2['{key}']: {value.shape}")
            
            print(f"\nForward pass successful!")
            print(f"Model can process robot action-image inputs correctly.")
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()

def test_action_embedding():
    """Test the action embedding module separately"""
    print("\nTesting Action Embedding Module")
    print("-" * 30)
    
    from dust3r.model_robot import ActionEmbedding
    
    action_dim = 7
    embed_dim = 768
    patch_size = 16
    img_size = (512, 288)
    batch_size = 2
    
    action_embed = ActionEmbedding(
        action_dim=action_dim,
        embed_dim=embed_dim,
        patch_size=patch_size,
        img_size=img_size
    )
    
    # Create sample actions
    actions = torch.randn(batch_size, action_dim)
    print(f"Input actions shape: {actions.shape}")
    
    # Forward pass
    action_features, action_pos = action_embed(actions)
    print(f"Action features shape: {action_features.shape}")
    print(f"Action positions shape: {action_pos.shape}")
    
    # Calculate expected number of patches
    num_patches_h = img_size[1] // patch_size
    num_patches_w = img_size[0] // patch_size
    expected_patches = num_patches_h * num_patches_w
    
    print(f"Expected patches: {expected_patches}")
    print(f"Actual patches: {action_features.shape[1]}")
    print(f"Match: {action_features.shape[1] == expected_patches}")

if __name__ == "__main__":
    # Test action embedding
    test_action_embedding()
    
    # Test full model
    main()
