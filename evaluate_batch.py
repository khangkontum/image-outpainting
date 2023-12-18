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


def crop_center(im: cv2.Mat, ratio: float = 0.2) -> cv2.Mat:
    x, y, _ = im.shape
    im = im[int(x * ratio) : x - int(x * ratio), int(y * ratio) : y - int(y * ratio)]
    return im


def evaluate(real_images, fake_images) -> float:
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    return fid.compute()


if __name__ == "__main__":
    # Load model
    input_dir = "./data/eval/wiki"
    model = load_model("./ckpts-wikiart-noadv/G_152.pt")
    html_path = "./eval-html-WN/0/imgs"

    for root, _, files in os.walk(input_dir):
        for index, file in tqdm(enumerate(files)):
            if ".png" in file:
                continue

            file_path = os.path.join(root, file)
            im = cv2.imread(file_path)
            try:
                cropped = crop_center(im)
            except:
                continue

            output_img, blended_img = perform_outpaint(model, cropped)
            blended_img = cv2.resize(blended_img, (512, 512))
            im = cv2.resize(im, (512, 512))

            folder_path = os.path.join(html_path, str(index))
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            cv2.imwrite(
                os.path.join(folder_path, file.replace(".jpg", "_truth.jpg")), im
            )
            cv2.imwrite(
                os.path.join(folder_path, file.replace(".jpg", "_gen.jpg")),
                blended_img * 255,
            )
            cv2.imwrite(
                os.path.join(folder_path, file.replace(".jpg", "_masked.jpg")), cropped
            )
