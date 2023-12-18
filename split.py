import os
import random
import shutil
from tqdm import tqdm

# Set paths
data_folder = './data/wikiart'
train_folder = './data/wiki_train_folder'
val_folder = './data/wiki_val_folder'
test_folder = './data/wiki_test_folder'

# Set random seed for reproducibility
random.seed(42)

# Create destination folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# List all image file names in the data folder
image_files = []
for root, subs, files  in os.walk(data_folder):
    for sub in subs:
        for file in os.listdir(os.path.join(root, sub)):
            if os.path.isfile(os.path.join(root, sub, file)):
                image_files.append(os.path.join(root, sub, file))

# Shuffle the image file names randomly
random.shuffle(image_files)
#
# # Split the image files into train, val, and test sets
train_split = 0.7  # Percentage of images for training
val_split = 0.15  # Percentage of images for validation
total_images = len(image_files)
train_count = int(total_images * train_split)
val_count = int(total_images * val_split)

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# Move files to respective folders

for file in tqdm(train_files):
    shutil.move(file, os.path.join(train_folder, file.split("/")[-1]))

for file in tqdm(val_files):
    shutil.move(file, os.path.join(val_folder, file.split("/")[-1]))

for file in tqdm(test_files):
    shutil.move(file, os.path.join(test_folder, file.split("/")[-1]))

print('Data split completed successfully.')
