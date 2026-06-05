import os
import base64
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader
import tqdm
from config.model_config import build_args
from net_utils import set_random_seed
from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

domain = 2
print("domain", domain)
df = pd.read_csv(f'llm_data/DomainNet_target_domain{domain}_4o-mini_v8.csv')

gt_unknown = df['private']
pred_unknown = df['unknown']

cmx = confusion_matrix(gt_unknown, pred_unknown)
cm_display = ConfusionMatrixDisplay(cmx)
cm_display.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix (Real Domain)')
# plt.savefig(f"target_{domain}_cm.png")
# df.to_csv(os.path.join("target_domain{}.csv".format(args.t_idx)), index=False)
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

def process_row(row):
    if row['predicted class name'][0] == '\'' and row['predicted class name'][-1] == '\'':
        return row['predicted class name'][1:-1]
    else:
        return row['predicted class name']

known = df[(df['private']==False)]

known['predicted class name'] = known.apply(process_row, axis=1)
labeling_accuracy = len(known[known['ground truth'] == known['predicted class name']]) / len(known)
print("known data labeling_accuracy", labeling_accuracy)

# confusion matrix
# calculate the percentage of true/true (1/1), false/false (0/0) (ground truth unknown/prediction unknown)
# df[private]
# for false/false, are the predicted labels same as the ground truths? what is the incorrect rate? (filter out the partition and save as a new csv file so that it is easier to view)
# among the incorrect predictions, are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?
tn = df[(df['private']==False) & (df['unknown']==False)]

tn['predicted class name'] = tn.apply(process_row, axis=1)
tn.to_csv(f'target_domain{domain}_tn.csv')
tn_filtered_notin = tn[~tn['predicted class name'].isin(class_list)]
tn_filtered_notin.to_csv(f'target_domain{domain}_tn_prediction_notinlist.csv')
tn_filtered_in = tn[tn['predicted class name'].isin(class_list)]
tn_filtered_in.to_csv(f'target_domain{domain}_tn_prediction_inlist.csv')
labeling_accuracy = len(tn[tn['ground truth'] == tn['predicted class name']]) / len(tn)
print("tn labeling_accuracy", labeling_accuracy)

label_out_perc = len(tn_filtered_notin) / len(tn)
print("tn label_out_perc", label_out_perc)
print("tn total", len(tn))

# for true/true, do the predicted labels make sense? what kind of logic is followed for the prediction? Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?
tp = df.loc[(df['private']==True) & (df['unknown']==True)]

tp['predicted class name'] = tp.apply(process_row, axis=1)
tp.to_csv(f'target_domain{domain}_tp.csv')
tp_filtered_notin = tp[~tp['predicted class name'].isin(class_list)]
tp_filtered_notin.to_csv(f'target_domain{domain}_tp_prediction_notinlist.csv')
tp_filtered_in = tp[tp['predicted class name'].isin(class_list)]
tp_filtered_in.to_csv(f'target_domain{domain}_tp_prediction_inlist.csv')

label_out_perc = len(tp_filtered_notin) / len(tp)
print("tp label_out_perc", label_out_perc)
print("tp total", len(tp))

# for false/true, are the predicted labels same as the ground truths? what is the incorrect rate? do the predicted labels make sense? what kind of logic is followed for the prediction? 
# Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?
fp = df.loc[(df['private']==False) & (df['unknown']==True)]

fp['predicted class name'] = fp.apply(process_row, axis=1)
fp.to_csv(f'target_domain{domain}_fp.csv')
fp_filtered_notin = fp[~fp['predicted class name'].isin(class_list)]
fp_filtered_notin.to_csv(f'target_domain{domain}_fp_prediction_notinlist.csv')
fp_filtered_in = fp[fp['predicted class name'].isin(class_list)]
fp_filtered_in.to_csv(f'target_domain{domain}_fp_prediction_inlist.csv')

labeling_accuracy = len(fp[fp['ground truth'] == fp['predicted class name']]) / len(fp)
print("fp labeling_accuracy", labeling_accuracy)

label_out_perc = len(fp_filtered_notin) / len(fp)
print("fp label_out_perc", label_out_perc)
print("fp total", len(fp))

# for true/false, do the predicted labels make sense? what kind of logic is followed for the prediction? Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?
fn = df.loc[(df['private']==True) & (df['unknown']==False)]

fn['predicted class name'] = fn.apply(process_row, axis=1)
fn.to_csv(f'target_domain{domain}_fn.csv')
fn_filtered_notin = fn[~fn['predicted class name'].isin(class_list)]
fn_filtered_notin.to_csv(f'target_domain{domain}_fn_prediction_notinlist.csv')
fn_filtered_in = fn[fn['predicted class name'].isin(class_list)]
fn_filtered_in.to_csv(f'target_domain{domain}_fn_prediction_inlist.csv')

label_out_perc = len(fn_filtered_notin) / len(fn)
print("fn label_out_perc", label_out_perc)
print("fn total", len(fn))