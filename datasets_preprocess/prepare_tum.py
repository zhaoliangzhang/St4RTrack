import glob
import os
import shutil
import numpy as np

def read_file_list(filename):
    """
    Reads a trajectory or file list from a text file.

    File format:
    "stamp d1 d2 d3 ..."

    Returns:
    dict -- dictionary of (stamp, [data]) pairs
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list_data = [[v.strip() for v in line.split(" ") if v.strip()!=""]
                 for line in lines
                 if len(line)>0 and line[0]!="#"]
    list_data = [(float(l[0]), l[1:]) for l in list_data if len(l) > 1]
    return dict(list_data)

def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp, data). As the time stamps never match exactly,
    we aim to find the closest match for every input tuple within 'max_difference'.
    """
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())

    potential_matches = []
    for a in first_keys:
        for b in second_keys:
            diff = abs(a - (b + offset))
            if diff < max_difference:
                potential_matches.append((diff, a, b))

    potential_matches.sort()  # Sort by time difference ascending

    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            # If both still available, pair them and remove from future matching
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    # Sort final matches by the first timestamp
    matches.sort(key=lambda x: x[0])
    return matches

def find_closest_depth_stamp(rgb_ts, depth_stamps, max_diff=0.02):
    """
    Given a single rgb_ts and a sorted array of depth_stamps,
    find the depth timestamp within 'max_diff' that is closest to 'rgb_ts'.
    Returns the matching depth timestamp or None if none found.
    """
    idx = np.searchsorted(depth_stamps, rgb_ts)
    candidates = []

    # We'll check the index right before and after 'idx' to see which is closer
    for i in [idx-1, idx, idx+1]:
        if 0 <= i < len(depth_stamps):
            diff = abs(rgb_ts - depth_stamps[i])
            if diff < max_diff:
                candidates.append((diff, depth_stamps[i]))

    if not candidates:
        return None
    # Return the depth timestamp that has the smallest difference
    return min(candidates, key=lambda x: x[0])[1]

dirs = glob.glob("../data/tum/*/")
dirs = sorted(dirs)

for dir in dirs:
    print(dir)
    # Paths to your text files
    rgb_file    = os.path.join(dir, 'rgb.txt')
    depth_file  = os.path.join(dir, 'depth.txt')
    gt_file     = os.path.join(dir, 'groundtruth.txt')

    if (not os.path.exists(rgb_file) or 
        not os.path.exists(depth_file) or
        not os.path.exists(gt_file)):
        print(f"Skipping {dir}, because one of rgb.txt, depth.txt, groundtruth.txt is missing.")
        continue

    # Read the file lists
    rgb_list   = read_file_list(rgb_file)       # stamp -> [path_to_rgb, ...]
    depth_list = read_file_list(depth_file)     # stamp -> [path_to_depth, ...]
    gt_list    = read_file_list(gt_file)        # stamp -> [Tx, Ty, Tz, Qx, Qy, Qz, Qw] (for instance)

    # First associate RGB with groundtruth
    matches_rgb_gt = associate(rgb_list, gt_list, offset=0.0, max_difference=0.02)

    # We will now find the *closest* depth for each matched (rgb, gt)
    # and only keep the triple if a depth image is found
    depth_stamps_sorted = np.array(sorted(depth_list.keys()))

    frames_rgb   = []
    frames_depth = []
    gt_data      = []

    for (rgb_ts, gt_ts) in matches_rgb_gt:
        # The actual RGB file name
        rgb_filename = os.path.join(dir, rgb_list[rgb_ts][0])

        # Find the depth timestamp that best matches rgb_ts
        d_ts = find_closest_depth_stamp(rgb_ts, depth_stamps_sorted, max_diff=0.02)
        if d_ts is None:
            # No suitable depth found, skip this pair
            continue

        depth_filename = os.path.join(dir, depth_list[d_ts][0])

        frames_rgb.append(rgb_filename)
        frames_depth.append(depth_filename)

        # groundtruth: store the time plus data
        gt_data.append([gt_ts] + gt_list[gt_ts])

    # Now, we sample 90 frames at stride 3
    frames_rgb   = frames_rgb[::3][:90]
    frames_depth = frames_depth[::3][:90]
    gt_data      = gt_data[::3][:90]

    # Make new dirs
    new_dir_rgb   = os.path.join(dir, 'rgb_90')
    new_dir_depth = os.path.join(dir, 'depth_90')
    os.makedirs(new_dir_rgb,   exist_ok=True)
    os.makedirs(new_dir_depth, exist_ok=True)

    # Copy the selected frames
    for rgb_path, depth_path in zip(frames_rgb, frames_depth):
        shutil.copy(rgb_path, new_dir_rgb)
        shutil.copy(depth_path, new_dir_depth)

    # Write out groundtruth for these 90 frames
    with open(os.path.join(dir, 'groundtruth_90.txt'), 'w') as f:
        for pose in gt_data:
            f.write(" ".join(map(str, pose)) + "\n")

    print(f"Done processing {dir}")
