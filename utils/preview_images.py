import matplotlib.pyplot as plt
from PIL import Image

# 原始資料 Bus：7293張
# 原始資料 Car：5095張


def check_loop_pic_datasets(kind):
    qa_list = []
    for id_ in range(1, 7293):
        if id_ < 100:
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
    check_loop_pic_datasets(kind='Car')