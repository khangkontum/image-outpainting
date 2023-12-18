from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from outpainting import perform_outpaint, load_model
from PIL import Image

import cv2
import torch
import numpy as np
import os
import sys


def postprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0

    return image

def crop_center(im: cv2.Mat, ratio: float = 0.2)->cv2.Mat:
    x, y, _ = im.shape
    im = im[int(x*ratio):x-int(x*ratio),int(y*ratio):y-int(y*ratio)]
    return im

def evaluate(real_images, fake_images)->float:
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    return fid.compute()


if __name__ == "__main__":
    input_dir = sys.argv[1]

    # Load model
    model_path = None
    if len(sys.argv) < 3:
        model_path = "./ckpts-wikiart-noadv/G_152.pt"
    else:
        model_path = sys.argv[2]
    model = load_model(model_path)

    real_images = []
    fake_images = []

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files[:500]):
            if ".png" in file:
                continue

            file_path = os.path.join(root, file) 
            im = cv2.imread(file_path)
            cropped = crop_center(im)

            output_img, blended_img = perform_outpaint(model, cropped)
            blended_img = cv2.resize(blended_img, (512, 512))
            im = cv2.resize(im, (512, 512))
            fake_images.append(blended_img)
            real_images.append(im)

            # cv2.imshow(file_path, blended_img)
            # cv2.imshow("cropped", cropped)
            # cv2.imshow("original", im)
            # cv2.imshow("output", output_img)
            # cv2.waitKey()
            # cv2.close()
            # exit(0)
    #         blended_img = preprocess_image(blended_img)
    fake_images = np.array(fake_images)
    real_images = np.array(real_images)
    #
    fake_images = np.transpose(fake_images, (0, 3, 1, 2))
    real_images = np.transpose(real_images, (0, 3, 1, 2))

    fake_images = torch.tensor(fake_images)
    real_images = torch.tensor(real_images) 

    print("FID score:", evaluate(real_images, fake_images))
    #         blended_img = postprocess_image(blended_img)
    #
    #         fake_images.append(preprocess_image(blended_img.numpy()))
    #         im = postprocess_image(im)
    #         real_images.append(im)
    #
    # real_images = torch.cat(real_images)
    # fake_images = torch.cat(fake_images) 
    #
        
