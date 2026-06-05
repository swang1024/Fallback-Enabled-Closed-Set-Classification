import csv

model_name = "4o-mini"
print(model_name)
csv_path = f"/hpc/group/carin/sw361/ChatGPT_exp/llm_data/DomainNet_target_domain0_{model_name}_v13_summary_pred.csv"
target_gt = "bottlecap"
max_rows = 4000

with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 0
    for row in reader:
        if count >= max_rows:
            break
        if row['ground truth'] == target_gt:
            print(row['predicted class name'])
        count += 1

print("----------------")

csv_path = f"/hpc/group/carin/sw361/ChatGPT_exp/llm_data/DomainNet_target_domain0_{model_name}_v8.csv"
target_gt = "bottlecap"
max_rows = 4000

with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 0
    for row in reader:
        if count >= max_rows:
            break
        if row['ground truth'] == target_gt:
            print(row['predicted class name'])
        count += 1
