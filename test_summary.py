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

df_direct = pd.read_csv(f"target_domain1_4o-mini_v8.csv", index_col=False)
def process_row(row):
    if row['predicted class name'][0] == '\'' and row['predicted class name'][-1] == '\'':
        return row['predicted class name'][1:-1]
    else:
        return row['predicted class name']
df_direct['predicted class name'] = df_direct.apply(process_row, axis=1)
llm_preds = list(df_direct['predicted class name'].values)
llm_preds_w_context = ["An image of " + cls for cls in llm_preds]

class_list = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', \
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
'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail']

user_promptv10 = "Can you give a summary of the image?"

# system_prompt1 = "You are an advanced AI that captions images, the caption should \
# try to look for visual features in the given category's description."

columns = ["idx", "ground truth", "private", "summary", "img url"]
 
data = []

model_name = "gpt-4o-mini"

partition = round(0.1 * len(target_train_dataloader))
print("number of samples", partition)

label_desc_df = pd.read_csv(f'label_visual_features.csv')
label_desc = {}
for cls in class_list:
    label_desc[cls] = label_desc_df[label_desc_df['class label'] == cls]['description'].values[0]

print(label_desc)

def lbl_desc(lbl):
    if lbl in class_list:
        return label_desc[lbl]
    else:
        return ""
    
# for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):
#     # only the first 10% of data
#     if idx >= 500:
#         break
    
#     # Getting the Base64 string
#     base64_image = encode_image(list(imgs_train)[0])

#     llm_preds_idx = llm_preds[idx]
#     visual_feature_desc = lbl_desc(llm_preds_idx)

#     # user_promptv12 = "Given that the prediction of this image from LLM is " + llm_preds_idx + ", \
#     #  think carefully why that might be then give a short summary of the image."

#     # user_promptv11 = "Given that the prediction of this image from LLM is " + llm_preds_idx + ", \
#     #  if you think it is true then give a short summary of the image, otherwise give a short summary \
#     # why the prediction and the image are misaligned."
    
#     # user_promptv13 = llm_preds_idx + "'s description is" + visual_feature_desc + " " + ""
#     # "Please give a short summary of the image"

#     user_promptv14 = "What are the useful visual features you can use to distinguish the category name of this photo?"

#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": user_promptv14},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{base64_image}",
#                             "detail":"low"},
#                     },
#                 ],
#             }
#         ],
#         max_completion_tokens=300
#     )
#     res = response.choices[0].message.content

#     print(imgs_idx.cpu().numpy()[0], list(ground_truth)[0])
#     print(private.cpu().numpy()[0])
#     print(list(imgs_train)[0])
#     print(res)

#     data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], private.cpu().numpy()[0], res, list(imgs_train)[0]])
#     df = pd.DataFrame(data, columns=columns)
#     df.to_csv(os.path.join("target_domain{}_4o-mini_v14_summary.csv".format(args.t_idx)), index=False)

columns = ["class label", "description"]

data = []

for idx, pred in enumerate(llm_preds[:500]):
    # only the first 10% of data
    if idx >= 500:
        break
    
    llm_preds_idx = llm_preds[idx]

    # user_promptv12 = "Given that the prediction of this image from LLM is " + llm_preds_idx + ", \
    #  think carefully why that might be then give a short summary of the image."

    # user_promptv11 = "Given that the prediction of this image from LLM is " + llm_preds_idx + ", \
    #  if you think it is true then give a short summary of the image, otherwise give a short summary \
    # why the prediction and the image are misaligned."
    
    # user_promptv13 = llm_preds_idx + "'s description is" + visual_feature_desc + " " + ""
    # "Please give a short summary of the image"

    user_promptv14 = "What are the useful visual features you can use to distinguish the category name in a photo?"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": user_promptv14},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": llm_preds_idx},
                    
                ],
            }
        ],
        max_completion_tokens=300
    )
    res = response.choices[0].message.content

    print(res)

    data.append([llm_preds_idx, res])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join("llm_pred_visual_features.csv".format(args.t_idx)), index=False)



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