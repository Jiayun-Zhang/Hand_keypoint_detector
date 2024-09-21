# Hand Keypoint detector

Create a virtual environment python 3.10:
'''
py -3.10 -m venv venv
.\venv\Scripts\activate
'''
or
'''
source venv/bin/activate
'''

Install the the dependencies. Here we use CUDA 11.7.
'''
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
'''

If install detectron2 failed, I tried to downgrade pip to 22.3 and it works
'''bash
pip install pip==22.3.1   
pip install git+https://github.com/facebookresearch/detectron2.git             
'''

Download the trained model
'''bash
bash fetch_demo_data.sh
'''

Run the code:
'''bash
python run.py --img_folder C:/Users/Jiayun/Desktop/hamer/demos_new/take1/rgb --out_folder keypoint_all_take1.json --batch_size=48 --side_view --save_mesh --full_frame
'''
