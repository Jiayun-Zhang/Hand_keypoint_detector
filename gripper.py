import json

json_file = "corrected_empty-vase_keypoint_all_take2.json"
with open(json_file, 'r') as f:
    data = json.load(f)

rgb_frames = [key for key in data.keys() if key.startswith("rgb_frame")]

for frame in rgb_frames:
    data[frame]["left"]["gripper"] = 1
    data[frame]["right"]["gripper"] = 1
    # print(data[frame]["left"]["gripper"])

rgb_frames_sorted = sorted(rgb_frames, key=lambda x: int(x[10:-4]))
# take1 100:490
# take2 [93:375] [82:375]
# take3 90:410
# take4 80:345
# take5 85:370
# take6 80:345
# take7 65:320
# take8 70:300

left_open_close_ranges = [
    [65, 203],   # left first close
#    [180, 220],  # left second close
    # can add more
]

right_open_close_ranges = [
    [115, 203],   # right first close
#    [180, 220],  # right second close
]

for start_idx, end_idx in left_open_close_ranges:
    for frame in rgb_frames_sorted[start_idx:end_idx+1]:
        data[frame]["left"]["gripper"] = 0
for start_idx, end_idx in right_open_close_ranges:
    for frame in rgb_frames_sorted[start_idx:end_idx+1]:
        data[frame]["right"]["gripper"] = 0

with open(json_file, 'w') as json_file:
    json.dump(data, json_file)