from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('image_path', type=str, help='Path to the image to be processed')

args = parser.parse_args()

model = YOLO("./weights/playing_cards.pt")

results = model(args.image_path)

#%%
