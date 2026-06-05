import os
import base64
from openai import OpenAI
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
from pydantic import BaseModel
import random

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

set_random_seed(2025)
print(args.target_data_dir)
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
random_seed = 0
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


user_promptv10 = "Can you give a summary of the image?"

user_promptv11 = "What is the main object of the image?"

user_promptv12 = "Describe the main object of the image."

user_promptv13 = "Identify the primary object in the image, excluding any background elements."

columns = ["idx", "ground truth", "private", "summary", "img url"]
 
data = []
model_name = "gpt-4o-mini"

for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(tqdm.tqdm(target_train_dataloader)):
    # if idx <= 12384:
    #     continue
    # if idx <= 11556:
    #     continue
    # if idx <= 12203 + 14914 - 1:
    #     continue

    # if idx <= 1000:
    #     continue
    # if idx > 4000:
    #     break
    # Getting the Base64 string
    base64_image = encode_image(list(imgs_train)[0])

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_promptv13},
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

    data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], private.cpu().numpy()[0], res, list(imgs_train)[0]])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join("llm_data/{}_{}_target_domain{}_4o-mini_summary_v13_randomseed{}.csv".format(target_type, args.dataset, args.t_idx, random_seed)), index=False)
