import numpy as np
from sklearn.cluster import KMeans
import os

# 讀取標籤文件，提取 bbox 寬度和高度


def load_data(label_path):
    bboxes = []
    for label_file in os.listdir(label_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_path, label_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    _, _, _, w, h = map(float, line.split())
                    bboxes.append([w, h])
    return np.array(bboxes)


# 使用 K-means 計算 anchors
def calculate_anchors(bboxes, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(bboxes)
    anchors = kmeans.cluster_centers_
    anchors = np.round(anchors * 640).astype(int)  # 調整到圖片尺度
    return anchors

# 執行計算
label_path = "../datasets/Transportation-train/labels"
bboxes = load_data(label_path)
num_clusters = 9  # 分為 9 組 anchors
anchors = calculate_anchors(bboxes, num_clusters)
print("New anchors are:", anchors)
