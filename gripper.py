import json

json_file = "corrected_aligned_keypoint_take8.json"
with open(json_file, 'r') as f:
    data = json.load(f)

rgb_frames = [key for key in data.keys() if key.startswith("rgb_frame")]

for frame in rgb_frames:
    data[frame]["left"]["gripper"] = 1  # 添加 "gripper":1
    data[frame]["right"]["gripper"] = 1  # 添加 "gripper":1
    # print(data[frame]["left"]["gripper"])  # 打印修改后的数据

rgb_frames_sorted = sorted(rgb_frames, key=lambda x: int(x[10:-4]))
# take1 100:490
# take2 [93:375] [82:375]
# take3 90:410
# take4 80:345
# take5 85:370
# take6 80:345
# take7 65:320
# take8 70:300
left_frames = rgb_frames_sorted[70:300]
right_frames = rgb_frames_sorted[70:300]
for frame in left_frames:
    data[frame]["left"]["gripper"] = 0  # 添加 "gripper":1
    # print(data[frame]["left"]["gripper"])  # 打印修改后的数据
for frame in right_frames:
    data[frame]["right"]["gripper"] = 0  # 添加 "gripper":1
    # print(data[frame]["left"]["gripper"])  # 打印修改后的数据
for frame in rgb_frames:
    print(frame)
    print(data[frame]["left"]["gripper"])

with open(json_file, 'w') as json_file:
    json.dump(data, json_file)