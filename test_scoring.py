import os
import base64
from openai import OpenAI
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader
import tqdm
from config.model_config import build_args
from net_utils import set_random_seed
from pathlib import Path
import pandas as pd
import json
from pydantic import BaseModel

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

args = build_args()
set_random_seed(2025)
print(args.target_data_dir)
target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    
target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
print(len(target_train_dataloader))

# user_promptv10 = "Can you give a summary of the image?"
system_promptv8 = "You are an AI that classifies images based on a predefined list of categories. \
If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."
user_promptv8 = "Does this image belong to one of the categories in the following list given the image and the text summary? \
['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', \
'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', \
'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', \
'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', \
'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', \
'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', \
'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', \
'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', \
'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', \
'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', \
'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', \
'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', \
'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', \
'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer',\
'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', \
'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', \
'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', \
'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', \
'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', \
'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', \
'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', \
'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail'] \
Please format the answer csv format with keys unknown and class_name separated by ',' \
Example 1:\
Image: (picture of a cat)\
Response: unknown: False, class_name: 'cat'\
Example 2:\
Image: (picture of a trumpet mouthpiece)\
Response: unknown: True, class_name: 'clarinet'\
"

columns = ["idx", "ground truth", "private", "summary", "img url"]
 
data = []

model_name = "gpt-4o-mini"

partition = round(0.1 * len(target_train_dataloader))
print("number of samples", partition)

for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):
    # only the first 10% of data
    if idx >= partition:
        break
    
    # Getting the Base64 string
    base64_image = encode_image(list(imgs_train)[0])

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_promptv8},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_promptv8},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail":"low"},
                    },
                ],
            }
        ],
        max_completion_tokens=300
    )
    res = response.choices[0].message.content

    print(imgs_idx.cpu().numpy()[0], list(ground_truth)[0])
    print(private.cpu().numpy()[0])
    print(list(imgs_train)[0])
    print(res)

    data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], private.cpu().numpy()[0], res, list(imgs_train)[0]])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join("target_domain{}_4o-mini_v10_summary.csv".format(args.t_idx)), index=False)


# df = pd.DataFrame(data, columns=columns)
# df.to_csv(os.path.join("target_domain{}.csv".format(args.t_idx)), index=False)

#  confusion matrix
# calculate the percentage of true/true, false/false (ground truth unknown/prediction unknown)
# df[private]
# for false/false, are the predicted labels same as the ground truths? what is the incorrect rate? 
# among the incorrect predictions, are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?
# for true/true, do the predicted labels make sense? what kind of logic is followed for the prediction? Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?

# for true/false, are the predicted labels same as the ground truths? what is the incorrect rate? do the predicted labels make sense? what kind of logic is followed for the prediction? Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?
# for false/true, do the predicted labels make sense? what kind of logic is followed for the prediction? Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?