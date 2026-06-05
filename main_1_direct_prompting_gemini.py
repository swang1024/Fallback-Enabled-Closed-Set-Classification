import os
import base64
from openai import OpenAI
from google import genai
from google.genai import types
from dataset.dataset import SFUniDADataset, INaturalist_UniDA
from torchvision.datasets import INaturalist
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import tqdm
from config.model_config import build_args
from net_utils import set_random_seed
from pathlib import Path
import pandas as pd
import json
import random
import time
from pydantic import BaseModel
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import os, certifi

# Nuke bad values (common on clusters)
for k in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(k, None)

# Force Python/httpx/OpenSSL to a known-good CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

args = build_args()

dataset = "INaturalist"

set_random_seed(2025)

if dataset == "DomainNet":
    args.dataset = "DomainNet"
    user_promptv8 = "Does this image belong to one of the categories in the following list \
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
    'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail']? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1:\
    Image: (picture of a cat)\
    Response: unknown: False, class_name: 'cat'\
    Example 2:\
    Image: (picture of a trumpet mouthpiece)\
    Response: unknown: True, class_name: 'clarinet'\
    "
    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)

    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
elif dataset == "VisDA":
    args.dataset = "VisDA"
    args.target_data_dir = "/hpc/group/carin/sw361/data/VisDA/validation/"
    user_promptv8 = "Does this image belong to one of the categories in the following list \
    ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1:\
    Image: (picture of a aeroplane)\
    Response: unknown: False, class_name: 'aeroplane'\
    Example 2:\
    Image: (picture of a donkey)\
    Response: unknown: True, class_name: 'horse'\
    "
    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)

    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
elif dataset == 'INaturalist':
    version = "v8"
    target_type = 'class'
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
    ])
    # Load the validation dataset 
    dataset_ = INaturalist(
        root='/hpc/group/carin/sw361/data/',
        version='2021_valid',
        target_type=target_type,
        transform=transform,
        download=True,
    )
    # Find out how many phylum categories there are by inspecting one sample
    # We'll scan through increasing IDs until category_name errors out or repeats
    label_names_list = []
    label_ids_list = []
    i = 0
    while True:
        try:
            name = dataset_.category_name(target_type, i)
            label_names_list.append(name)
            label_ids_list.append(i)
            i += 1
        except (IndexError, ValueError):
            break

    print(f"Total label categories: {len(label_names_list)}")
    print("label categories:", label_names_list, label_ids_list)

    # randomly split the label_ids_list into two
    random_seed = 2
    print("seed", random_seed)
    random.Random(random_seed).shuffle(label_ids_list)
    if target_type == 'phylum':
        shared_class_num = 5
        source_private_class_num = 4
    elif target_type == 'class':
        shared_class_num = 20
        source_private_class_num = 15
    target_private_class_num = len(label_ids_list) - shared_class_num - source_private_class_num
    shared_class_ids_list = label_ids_list[:shared_class_num]
    source_private_class_ids_list = label_ids_list[shared_class_num:shared_class_num+source_private_class_num]
    target_private_class_ids_list = label_ids_list[shared_class_num+source_private_class_num:]
    source_classes = shared_class_ids_list + source_private_class_ids_list
    target_classes = shared_class_ids_list + target_private_class_ids_list
    known_class_list = [label_names_list[i] for i in source_classes]
    unknown_class_list = [label_names_list[i] for i in target_classes]

    user_promptv8 = f"Does this image belong to one of the categories in the following list \
    {known_class_list}? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1: \
    Image: (picture of a aeroplane) \
    Response: unknown: False, class_name: 'aeroplane' \
    Example 2: \
    Image: (picture of a donkey) \
    Response: unknown: True, class_name: 'horse' \
    # "
    print("known_class_list", known_class_list, flush=True)

    target_dataset = INaturalist_UniDA(root='/hpc/group/carin/sw361/data/', version='2021_valid', target_type=target_type, transform=transform, download=True, shared_classes=shared_class_ids_list, source_private_classes=source_private_class_ids_list, target_private_classes=target_private_class_ids_list, label_names_list=label_names_list)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)

print(len(target_train_dataloader))

version = "v8"
system_promptv8 = "You are an AI that classifies images based on a predefined list of categories. \
If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."

columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

data = []

model_name = "gemini-2.0-flash"

# partition = round(0.1 * len(target_train_dataloader))
# print("number of samples", partition)
# df1 = pd.read_csv(f'VisDA_target_domain1_4o-mini_v8.csv')
# img_urls = list(df1['img url'].values)
def safe_generate(model_name, image_bytes, system_prompt, user_prompt):
    retries = 5
    for i in range(retries):
        try:
            return client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt),
                contents=[types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ), user_prompt]
            )
        except genai.errors.ServerError as e:
            if i == retries - 1:
                raise
            sleep_time = 2 ** i  # exponential backoff
            print(f"Model overloaded, retrying in {sleep_time}s...")
            time.sleep(sleep_time)

for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):
    # if idx < 23892:
    #     continue
    # if imgs_train not in img_urls:
    #     print(imgs_train)
    # if idx < 690:
    #     continue
    # if idx > 10000:
    #     break

    # Getting the Base64 string
    # base64_image = encode_image(list(imgs_train)[0])
    # print(list(imgs_train)[0])
    with open(list(imgs_train)[0], 'rb') as f:
        image_bytes = f.read()

    response = safe_generate(model_name, image_bytes, system_promptv8, user_promptv8)
    res = response.text
    # print(res)
    # try to extract class_name and unknown
    unknown = ""
    class_name = ""
    try:
        parts = res.split(',')
        for p in parts:
            if 'unknown' in p:
                unknown = p.split('unknown:')[-1].strip()
            if 'class_name' in p:
                class_name = p.split('class_name:')[-1].strip().strip("'\" ")
    except Exception:
        pass

    data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], class_name, private.cpu().numpy()[0], unknown, list(imgs_train)[0]])
    df = pd.DataFrame(data, columns=columns)
    
    if dataset == "DomainNet" or dataset == "VisDA":
        df.to_csv(os.path.join("llm_data/{}_target_domain{}_{}_{}.csv".format(dataset, args.t_idx, model_name, version)), index=False)
    elif dataset == "INaturalist":
        df.to_csv(os.path.join("llm_data/{}_{}_target_domain{}_{}_{}.csv".format(target_type, dataset, args.t_idx, model_name, version)), index=False)
