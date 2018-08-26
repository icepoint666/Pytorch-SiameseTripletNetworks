"""Split the dataset into train/val and resize the images to HEIGHT x WIDTH"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

HEIGHT = 64
WIDTH = 64

TRAIN_PROPORTION = 0.8

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/', help="Directory with the specified dataset")
parser.add_argument('--output_dir', default='data/', help="Where to write the new data")
parser.add_argument('-p', '--train_proportion', default=TRAIN_PROPORTION, type=float, help="Proportion of training data")
parser.add_argument('--resize', default='True', help="Whether to resize the images")
parser.add_argument('-h', '--height', default=HEIGHT, type=int, help="Height of resized images")
parser.add_argument('-w', '--width', default=WIDTH, type=int, help="Width of resized images")

def resize_and_save(filename, output_dir, height=HEIGHT, width=WIDTH):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((height, width), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))

def save(filename, output_dir):
    """Save it to the `output_dir`"""
    image = Image.open(filename)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in data directory
    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames if (f.endswith('.jpg')) or (f.endswith('.png'))]

    # Split the images in train into 'train' and 'val'
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(args.train_proportion * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames}
    # Preprocess train, val
    for split in ['train', 'val']:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            if args.resize == 'True':
                resize_and_save(filename, output_dir_split, args.height, args.width)
            else:
                save(filename, output_dir_split)

    print("Done building dataset")
