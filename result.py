import torch

import pandas as pd
import csv

from torchvision.datasets import ImageFolder

import torch.nn as nn, torch.nn.functional as F
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

file_path = '/root/FathomNet25/results_v4.pkl'
data = torch.load(file_path)
output_csv = 'pred_swin_pmg_v4.csv'
logits = data['logits']
p_names = data['p_names']
img_path = data['im_paths']

zipped = zip(logits, p_names, img_path)


def get_index_from_filename(filename):
    # 拆分文件名，提取 _ 后的数字（去掉扩展名）
    base = os.path.splitext(filename)[0]  # 去掉扩展名
    try:
        index = int(base.split('_')[-1])  # 提取最后的数字部分
    except ValueError:
        index = float('inf')  # 如果不是数字，排在最后
    return index

sorted_data = sorted(zipped, key=lambda item: get_index_from_filename(item[2]))
train_dataset = ImageFolder('/root/autodl-tmp/dataset/train')
class_names = train_dataset.classes
def gen_cost_matrix(file_path):
    cost_df = pd.read_csv(file_path, index_col=0)  # 替换为实际路径
    cost_matrix_np = cost_df.values.astype(float)  # 确保数值为浮点型
    cost_matrix = torch.from_numpy(cost_matrix_np).float()
    return cost_matrix

cost_matrix = gen_cost_matrix("/root/FathomNet25/cost_metrix.csv").to(DEVICE)


num_samples = len(logits)
print(num_samples)
print(len(p_names))
print(len(img_path))
results = []
id = 1

for cur_logits, cur_p_name, cur_img_path in sorted_data:
    
    prob = F.softmax(cur_logits)
    expected_cost = prob @ cost_matrix
    predicted_label = class_names[expected_cost.argmin(dim=0).item()]
    #print(cur_img_path)
    #print(cur_p_name)
    results.append({"annotation_id": id, "concept_name": predicted_label})
    #results.append({"annotation_id": id, "concept_name": cur_p_name})
    id += 1
    
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"推理完成，结果已保存至 {output_csv}")
    
    
    