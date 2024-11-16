import time

import matplotlib.pyplot as plt
from PIL import Image

# 原始資料 Bus：7293張
# 原始資料 Car：5095張


def check_loop_pic_datasets(kind):
    qa_list = []
    for id_ in range(1, 7293):
        if id_ < 10:
            # id_ = 30
            image_path = f'../datasets/{kind}/images/{kind}_0{id_}.jpg'
            label_path = f'../datasets/{kind}/labels/{kind}_0{id_}.txt'
        else:
            image_path = f'../datasets/{kind}/images/{kind}_{id_}.jpg'
            label_path = f'../datasets/{kind}/labels/{kind}_{id_}.txt'

        # 讀取圖片
        image = Image.open(image_path)

        # 讀取標記資訊
        with open(label_path, 'r') as file:
            label_data = file.readlines()

        # 提取標記資訊
        for line in label_data:
            parts = line.strip().split()
            cls = 0  # 類別
            x_center = float(parts[1])  # 標記框中心 x
            y_center = float(parts[2])  # 標記框中心 y
            width = float(parts[3])  # 標記框寬度
            height = float(parts[4])  # 標記框高度

            # 計算左上角和右下角的座標
            img_width, img_height = image.size
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height

            # 繪製標記框
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, edgecolor='red', linewidth=2))

            # 顯示圖片
            plt.imshow(image)
            plt.axis('off')  # 不顯示坐標軸
            plt.show()
        qa = input("Is fined?")
        if qa != '':
            qa_list.append(id_)

    print(qa_list)


if __name__ == '__main__':
    check_loop_pic_datasets()
# failed_list = [2, 3, 5, 7, 9, 13, 16, 17, 19, 27, 29, 30, 31, 32, 40, 59, 60, 61, 64, 67, 68, 69, 70, 72, 73, 74, 77, 78, 79, 81, 82, 90, 91, 92, 101, 102, 103, 104, 108, 110, 112, 120, 122, 124, 125, 127, 130, 132, 138, 140, 142, 145, 147, 148, 151, 154, 155, 165, 169, 171, 172, 174, 175, 181, 187, 188, 189, 193, 197, 198, 200, 201, 203, 207, 210, 212, 214, 215, 216, 218, 219, 220, 222, 223, 226, 229, 230, 234, 236, 238, 239, 240, 243, 245, 247, 250, 254, 255, 257, 258, 259, 267, 270, 279, 280, 282, 283, 288, 290, 292, 293, 294, 297, 300, 301, 302, 309, 313, 318, 320, 322, 332, 333, 334, 338, 339, 342, 344, 346, 348, 349, 351, 354, 357, 358, 359, 361, 365, 368, 369, 370, 371, 378, 379, 380, 385, 387, 389, 390, 392, 393, 394, 395, 399, 409, 416, 418, 419, 423, 430, 435, 436, 441, 443, 444, 445, 446, 447, 448, 452, 454, 455, 459, 460, 461, 462, 463, 464, 465, 467, 468, 469, 470, 471, 472, 473, 478, 480, 484, 487, 488, 491, 493, 495, 496]