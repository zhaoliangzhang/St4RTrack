# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import glob

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ToTensor = tvf.ToTensor()
TAG_FLOAT = 202021.25

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def flow_read(filename):
    """ Read optical flow from file, return (U,V) tuple. 
    
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]
    return u,v

def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size, nearest=False):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS if not nearest else PIL.Image.NEAREST
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def create_patch_level_mask(image, mask, patch_size=16, threshold=0.95):

    """Create a patch level mask based on the number of masked pixels in each patch.
    input:
    image (torch.Tensor): [B, C, H, W] tensor representing the image
    mask (torch.Tensor): [B, H, W] tensor representing the mask
    patch_size (int): the size of the patches
    threshold (float): the threshold for the number of masked pixels in a patch
    """
    
    # Ensure the image and mask are PyTorch tensors
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    
    # Get the dimensions of the image
    height, width = image.shape[-2:]
    
    # Calculate the number of patches
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size
    
    # Initialize the patch level mask
    patch_mask = torch.zeros((num_patches_y, num_patches_x), dtype=torch.bool)
    
    # Calculate the threshold in terms of the number of pixels
    patch_area = patch_size * patch_size
    pixel_threshold = patch_area * threshold
    
    # Loop through each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Extract the patch from the mask
            patch = mask[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            # Calculate the number of masked pixels in the patch
            num_masked_pixels = torch.sum(patch)
            # Set the patch mask if the number of non-masked pixels exceeds the threshold
            if num_masked_pixels > pixel_threshold:
                patch_mask[i, j] = True # True means the patch is not masked
                
    return patch_mask

def crop_img(img, size, square_ok=False, nearest=False, crop=True):
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)), nearest=nearest)
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size, nearest=nearest)
    W, H = img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        if crop:
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        else: # resize
            img = img.resize((2*halfw, 2*halfh), PIL.Image.LANCZOS)
    return img

def load_images(folder_or_list, size, square_ok=False, verbose=True, dynamic_mask_root=None, crop=True, fps=0, num_frames=110, start_frame=0, step_size=1):
    """Open and convert all images or videos in a list or folder to proper input format for DUSt3R."""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        # if folder_or_list is a folder, load all images in the folder
        if os.path.isdir(folder_or_list):
            root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
        else: # the folder_content will be the folder_or_list itself
            root, folder_content = '', [folder_or_list]

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} items')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'Bad input {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    supported_video_extensions = ['.mp4', '.avi', '.mov']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)
    supported_video_extensions = tuple(supported_video_extensions)

    # Sort items by their names
    if isinstance(folder_content[0], str):
        folder_content = sorted(folder_content, key=lambda x: x.split('/')[-1])
    
    # Apply start_frame and step_size to the folder_content for non-video inputs
    if not any(path.lower().endswith(supported_video_extensions) for path in folder_content if isinstance(path, str)):
        folder_content = folder_content[start_frame:start_frame+num_frames:step_size]
        if verbose and (start_frame > 0 or step_size > 1):
            print(f' - Using start_frame={start_frame}, step_size={step_size}, selected {len(folder_content)} items')
    
    imgs = []
    for path in folder_content:
        if isinstance(path, str):
            full_path = os.path.join(root, path)

        # if file is PIL.Image
        if isinstance(path, PIL.Image.Image):
            img = exif_transpose(path).convert('RGB')
            W1, H1 = img.size
            img = crop_img(img, size, square_ok=square_ok, crop=crop)
            W2, H2 = img.size

            if verbose:
                print(f' - Adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
            
            single_dict = dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=None,
                mask=~(ToTensor(img)[None].sum(1) <= 0.01)
            )
            single_dict['dynamic_mask'] = torch.zeros_like(single_dict['mask'])
            imgs.append(single_dict) 

        elif path.lower().endswith(supported_images_extensions):
            # Process image files
            img = exif_transpose(PIL.Image.open(full_path)).convert('RGB')
            W1, H1 = img.size
            img = crop_img(img, size, square_ok=square_ok, crop=crop)
            W2, H2 = img.size

            if verbose:
                print(f' - Adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
            
            single_dict = dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=full_path,
                mask=~(ToTensor(img)[None].sum(1) <= 0.01)
            )
            
            if dynamic_mask_root is not None:
                dynamic_mask_path = os.path.join(dynamic_mask_root, os.path.basename(path))
            else:  # Sintel dataset handling
                dynamic_mask_path = full_path.replace('final', 'dynamic_label_perfect').replace('clean', 'dynamic_label_perfect')

            if os.path.exists(dynamic_mask_path):
                dynamic_mask = PIL.Image.open(dynamic_mask_path).convert('L')
                dynamic_mask = crop_img(dynamic_mask, size, square_ok=square_ok)
                dynamic_mask = ToTensor(dynamic_mask)[None].sum(1) > 0.99  # "1" means dynamic
                if dynamic_mask.sum() < 0.8 * dynamic_mask.numel():  # Consider static if over 80% is dynamic
                    single_dict['dynamic_mask'] = dynamic_mask
                else:
                    single_dict['dynamic_mask'] = torch.zeros_like(single_dict['mask'])
            else:
                single_dict['dynamic_mask'] = torch.zeros_like(single_dict['mask'])

            imgs.append(single_dict)

        elif path.lower().endswith(supported_video_extensions):
            # Process video files
            if verbose:
                print(f'>> Loading video from {full_path}')
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f'Error opening video file {full_path}')
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if video_fps == 0:
                print(f'Error: Video FPS is 0 for {full_path}')
                cap.release()
                continue
            if fps > 0:
                frame_interval = max(1, int(round(video_fps / fps)))
            else:
                frame_interval = 1
            frame_indices = list(range(0, total_frames, frame_interval))
            if num_frames is not None:
                frame_indices = frame_indices[start_frame:num_frames+start_frame:step_size]

            if verbose:
                print(f' - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}')

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                W1, H1 = img.size
                img = crop_img(img, size, square_ok=square_ok, crop=crop)
                W2, H2 = img.size

                if verbose:
                    print(f' - Adding frame {frame_idx} from {path} with resolution {W1}x{H1} --> {W2}x{H2}')
                
                single_dict = dict(
                    img=ImgNorm(img)[None],
                    true_shape=np.int32([img.size[::-1]]),
                    idx=len(imgs),
                    instance=f'{full_path}_frame_{frame_idx}',
                    mask=~(ToTensor(img)[None].sum(1) <= 0.01)
                )

                # Dynamic masks for video frames are set to zeros by default
                single_dict['dynamic_mask'] = torch.zeros_like(single_dict['mask'])

                imgs.append(single_dict)

            cap.release()
        
        else:
            continue  # Skip unsupported file types

    assert imgs, 'No images found at ' + root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def enlarge_seg_masks(folder, kernel_size=5):
    mask_pathes = glob.glob(f'{folder}/dynamic_mask_*.png')
    for mask_path in mask_pathes:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((kernel_size, kernel_size),np.uint8)
        enlarged_mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(mask_path.replace('dynamic_mask', 'enlarged_dynamic_mask'), enlarged_mask)