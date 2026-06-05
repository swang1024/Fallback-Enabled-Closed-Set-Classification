import os
import base64
from openai import OpenAI
from dataset.dataset import SFUniDADataset, INaturalist_UniDA
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import INaturalist
import tqdm
from config.model_config import build_args
from net_utils import set_random_seed
from pathlib import Path
import pandas as pd
import json
from pydantic import BaseModel
import torch.multiprocessing as mp
import random
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

dataset = "INaturalist"
args.dataset = "INaturalist"
version = "v13"

set_random_seed(2025)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
])
target_type = 'phylum'
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

target_dataset = INaturalist_UniDA(root='/hpc/group/carin/sw361/data/', version='2021_valid', target_type=target_type, transform=transform, download=True, shared_classes=shared_class_ids_list, source_private_classes=source_private_class_ids_list, target_private_classes=target_private_class_ids_list, label_names_list=label_names_list)
target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
print(len(target_train_dataloader))

system_promptv8 = "You are an AI that classifies images based on a summary of the image. \
If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."

user_promptv8 = "Does this image belong to one of the categories in the following list \
    {} based on the following summary: {}? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1: \
    Image: (picture of a aeroplane) \
    Response: unknown: False, class_name: 'aeroplane' \
    Example 2: \
    Image: (picture of a donkey) \
    Response: unknown: True, class_name: 'horse' \
    "
print("known_class_list", known_class_list, flush=True)

columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

data = []

df_summary = pd.read_csv(f'llm_data/{target_type}_{dataset}_target_domain{args.t_idx}_4o-mini_summary_{version}_randomseed{random_seed}.csv', index_col=False)

for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):
    # if idx < 23892:
    #     continue
    # if imgs_train not in img_urls:
    #     print(imgs_train)
    try:
        summary = df_summary[df_summary['idx'] == imgs_idx[0].cpu().numpy()]['summary'].values[0]
    except IndexError: 
        summary = df_summary[df_summary['idx'] == imgs_idx[0].cpu().numpy()]['summary'].values

    # if idx > 10:
    #     break
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
                        "text": user_promptv8.format(known_class_list, summary)},
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
    df.to_csv(os.path.join("llm_data/{}_{}_target_domain{}_{}_{}_summary_pred_randomseed{}.csv".format(target_type, dataset, args.t_idx, model_name, version, random_seed)), index=False)
