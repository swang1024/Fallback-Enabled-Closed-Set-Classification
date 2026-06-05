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

dataset = "DomainNet"
if dataset == "DomainNet":
    args.dataset = "DomainNet"
elif dataset == "VisDA":
    args.dataset = "VisDA"

set_random_seed(2025)
print(args.target_data_dir)
target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    
target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
print(len(target_train_dataloader))

user_promptv10 = "Can you give a summary of the image?"

user_promptv11 = "What is the main object of the image?"

user_promptv12 = "Describe the main object of the image."

user_promptv13 = "Identify the primary object in the image, excluding any background elements."

columns = ["idx", "ground truth", "private", "summary", "img url"]
 
data = []
model_name = "gpt-4o-mini"

for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):
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
    df.to_csv(os.path.join("llm_data/{}_target_domain{}_4o-mini_summary_v13.csv".format(args.dataset, args.t_idx)), index=False)
