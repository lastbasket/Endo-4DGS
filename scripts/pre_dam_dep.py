import os
from PIL import Image
import numpy as np
import onnxruntime as ort
from argparse import ArgumentParser


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
session = ort.InferenceSession(
            'submodules/depth_anything/weights/depth_anything_vits14.onnx', 
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )


def normalize_img(img, mean, std):
    img = (img - mean) / std
    return img

def get_depth(path):
    color = (np.array(Image.open(path))/255.0).astype(np.float32)

    norm_img = np.transpose(normalize_img(color, mean, std), (2, 0, 1))[None].astype(np.float32)
    depth_es = session.run(None, {"image": norm_img})[0][0]
    depth_es = 1 / depth_es * 1000
    return depth_es


def main(args):    
    img_dir = os.path.join(args.dataset_root, args.rgb_paths)
    dam_depth_dir = img_dir.replace(args.rgb_paths, 'depth_dam')
    os.makedirs(dam_depth_dir, exist_ok=True)

    img_list = os.listdir(img_dir)
    for i in img_list:
        img_path = os.path.join(img_dir, i)
        dam_depth_path = os.path.join(dam_depth_dir, i).replace('png', 'npy')
        dam_depth = get_depth(img_path)
        print(dam_depth_path)
        np.save(dam_depth_path, dam_depth)
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default='data/endonerf/pulling')
    parser.add_argument("--rgb_paths", type=str, default='images')
    args = parser.parse_args()
    main(args)