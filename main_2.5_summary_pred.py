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
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

model_name = "gpt-4o-mini"

if model_name == "gpt-4o-mini":
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
args = build_args()

dataset = "VisDA"

if dataset == "DomainNet":
    args.dataset = "DomainNet"
    version = "v13"
    user_promptv13 = "Does this image belong to one of the categories in the following list \
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
    'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail'] based on the follow summary: {}? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1:\
    Image: (picture of a cat)\
    Response: unknown: False, class_name: 'cat'\
    Example 2:\
    Image: (picture of a trumpet mouthpiece)\
    Response: unknown: True, class_name: 'clarinet'\
    "
elif dataset == "VisDA":
    args.dataset = "VisDA"
    version = "v10"
    user_promptv13 = "Does this image belong to one of the categories in the following list \
    ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant'] based on the follow summary: {}? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1:\
    Image: (picture of a aeroplane)\
    Response: unknown: False, class_name: 'aeroplane'\
    Example 2:\
    Image: (picture of a donkey)\
    Response: unknown: True, class_name: 'horse'\
    "
   
set_random_seed(2025)
print(args.target_data_dir)
target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)

target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
print(len(target_train_dataloader))

system_promptv8 = "You are an AI that classifies images based on a summary of the image. \
If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."

columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

data = []

df_summary = pd.read_csv(f'llm_data/{dataset}_target_domain{args.t_idx}_4o-mini_summary_{version}.csv', index_col=False)

for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):
    # if idx < 23892:
    #     continue
    # if imgs_train not in img_urls:
    #     print(imgs_train)
    try:
        summary = df_summary[df_summary['idx'] == imgs_idx[0].cpu().numpy()]['summary'].values[0]
    except IndexError: 
        summary = df_summary[df_summary['idx'] == imgs_idx[0].cpu().numpy()]['summary'].values

    # print(imgs_idx[0].cpu().numpy())
    # print(summary)
    # print(list(imgs_train)[0])

    if idx > 4000:
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
                        "text": user_promptv13.format(summary)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail":"low"},
                    },
                ],
            }
        ]
    )
    res = response.choices[0].message.content

    data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], res.split(', class_name: ')[1], private.cpu().numpy()[0], res.split(', class_name: ')[0].split('unknown: ')[1], list(imgs_train)[0]])
    df = pd.DataFrame(data, columns=columns)
    # df.to_csv(os.path.join("llm_data/{}_target_domain{}_{}_{}_summary_pred.csv".format(dataset, args.t_idx, model_name, version)), index=False)
