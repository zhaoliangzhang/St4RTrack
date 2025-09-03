import cv2
import torch
from moge.model import MoGeModel

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             

# Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
input_image = cv2.cvtColor(cv2.imread("example_images/BunnyCake.jpg"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer 
output = model.infer(input_image)
# `output` has keys "points", "depth", "mask" and "intrinsics",
# The maps are in the same size as the input image. 
# {
#     "points": (H, W, 3),    # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
#     "depth": (H, W),        # scale-invariant depth map
#     "mask": (H, W),         # a binary mask for valid pixels. 
#     "intrinsics": (3, 3),   # normalized camera intrinsics
# }