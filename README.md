# Hand Keypoint detector

Create a virtual environment python 3.10:


```
py -3.10 -m venv venv
.\venv\Scripts\activate
```
or

```
source venv/bin/activate
```

Install the the dependencies. Here we use CUDA 11.7.

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose

```

If install detectron2 failed, I tried to downgrade pip to 22.3 and it works

```
pip install pip==22.3.1   
pip install git+https://github.com/facebookresearch/detectron2.git             

```

Download the trained model

```
bash fetch_demo_data.sh

```

Run the code:

```
python run.py --img_folder C:/Users/Jiayun/Desktop/hamer/demos_new/take1/rgb --out_file keypoint_all_take1.json --batch_size=48 --full_frame

```
Correct the position of key points using the depth map
```
python correct_kp_bias.py --rgb_folder "C:/Users/Jiayun/Desktop/data/empty-vase_take2/rgb" --depth_folder "C:/Users/Jiayun/Desktop/data/empty-vase_take2/depth" --json_file "empty-vase_keypoint_all_take2.json"
# modify the rgb_folder, depth_folder and json_file arguments
```
Convert hand key points into gripper poses and generate visualization videos
```
python hand_to_gripper.py --rgb_folder "C:/Users/Jiayun/Desktop/data/empty-vase_take2/rgb" --depth_folder "C:/Users/Jiayun/Desktop/data/empty-vase_take2/depth" --json_file "corrected_empty-vase_keypoint_all_take2.json" 
# modify rgb_folder, depth_folder and json_file
```

Manually modify the opening or closing of the gripper
```
python gripper.py 
```
