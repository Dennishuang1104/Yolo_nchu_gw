from PIL import Image
from tqdm import tqdm
from yolov3_pytorch_modules.utils.general import os, Path
import shutil
import glob


class DataPreProcessing:
    def __init__(self, kind_list: list):
        self.kind_list = kind_list
        self.counts_data = 2000  # 拿500 張訓練
        self.img_type = 'jpg'
        self.dataset_path = 'datasets/'
        self.model_dataset_path = 'datasets/Transportation-train'
        self.class_dict = {
            "Car": 0,
            "Bus": 1,
            "Bike": 2
        }

    def get_filenames(self, kind):
        # step 1 把Label 內.txt資料搬出來
        txt_dir_path = f'{self.dataset_path}/{kind}/annotations'
        jpg_dir_path = f'{self.dataset_path}/{kind}/images'
        if not os.path.exists(jpg_dir_path):
            os.makedirs(jpg_dir_path)
        if not os.path.exists(txt_dir_path):
            os.makedirs(txt_dir_path)
        dir_path = f'../datasets/{kind}'
        txt_directory = os.path.join(os.path.dirname(__file__), txt_dir_path)
        for filename in os.listdir(txt_directory):
            original_path = os.path.join(txt_dir_path, filename)
            new_path = os.path.join(dir_path, filename)
            os.rename(original_path, new_path)

        # step 2 把對應ID 取代成
        id_list = []
        directory = os.path.join(os.path.dirname(__file__), dir_path)
        for idx, filename in enumerate(os.listdir(directory), start=1):
            if filename.endswith(f'.{self.img_type}'):
                id_list.append(filename.replace(f'.{self.img_type}', ''))

        # step 3 將ID list 元素存成新的/.txt, .jpg
        for idx, id_ in enumerate(id_list, start=1):
            original_jpg_name = f'{id_}.jpg'
            original_txt_name = f'{id_}.txt'
            jpg_filename = f"{kind}_{str(idx).zfill(2)}.jpg"
            txt_filename = f"{kind}_{str(idx).zfill(2)}.txt"

            original_txt_path = os.path.join(dir_path, original_txt_name)
            new_txt_path = os.path.join(txt_dir_path, txt_filename)
            os.rename(original_txt_path, new_txt_path)

            original_jpg_path = os.path.join(dir_path, original_jpg_name)
            new_jpg_path = os.path.join(jpg_dir_path, jpg_filename)
            os.rename(original_jpg_path, new_jpg_path)

    def located_datasets(self):
        for kind in self.kind_list:
            self.get_filenames(kind=kind)

    def convert_box(self, size, box):
        '''
        將絕對座標轉換為 YOLO 格式的相對座標。
        :param size: (width, height) 圖片尺寸
        :param box: [XMin, XMax, YMin, YMax] 絕對座標框
        :return: [class_id, x_center, y_center, width, height] YOLO 格式
        '''
        image_width, image_height = size
        x_min, y_min, x_max, y_max = box
        # 計算中心點和寬高的相對座標
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # 判斷是否有任何數值超過 1
        if x_center > 1 or y_center > 1 or width > 1 or height > 1:
            print("Error")

        return (x_center, y_center, width, height)

    def visdrone2yolo(self, kind_dir):
        full_kind_dir = Path(self.dataset_path + kind_dir)
        (full_kind_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
        # pbar = tqdm((full_kind_dir / 'annotations').glob('*.txt'), desc=f'Converting {kind_dir}')
        pbar = tqdm(
            [f for f in (full_kind_dir / 'annotations').glob(f'{kind_dir}_*.txt')
             if int(f.stem.split('_')[1]) < self.counts_data],
            desc=f'Converting {kind_dir}'
        )
        for img_file in pbar:
            img_size = Image.open((full_kind_dir / 'images' / img_file.name).with_suffix('.jpg')).size
            lines = []
            with open(img_file, 'r') as file:  # read annotation.txt
                for row in [x.split(' ') for x in file.read().strip().splitlines()]:
                    cls = self.class_dict[str((row[0]))]
                    box = self.convert_box(img_size, tuple(map(float, row[1:5])))
                    lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                    with open(str(img_file).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                        fl.writelines(lines)  # write label.txt

    def combine_and_split(self):
        """
        整併不同類別照片 to Transportation-train 這個目錄
        :return:
        """
        if not os.path.exists(self.model_dataset_path):
            os.makedirs(self.model_dataset_path, exist_ok=True)

        for category in self.class_dict.keys():
            # 定義 a 和 c 子目錄的路徑
            for sub_folder in ["annotations", "images", "labels"]:
                if not os.path.exists(self.model_dataset_path + f"/{sub_folder}"):
                    os.makedirs(self.model_dataset_path + f"/{sub_folder}")
                sub_dir = os.path.join(self.dataset_path, category, sub_folder)
                if os.path.isdir(sub_dir):
                    # 列出該目錄中的所有檔案
                    all_files = os.listdir(sub_dir)
                    for i in range(1, self.counts_data):
                        if i < 10:
                            target_file = f"{category}_0{i}.txt"
                        else:
                            target_file = f"{category}_{i}.txt"
                        if sub_folder == 'images':
                            target_file = target_file.replace(".txt", ".jpg")
                        if target_file in all_files:
                            source_path = os.path.join(sub_dir, target_file)
                            # 目標檔案路徑
                            dest_path = os.path.join(self.model_dataset_path+f"/{sub_folder}", target_file)
                            # 複製檔案
                            shutil.copy(source_path, dest_path)

    def clean_dirty_data(self):
        bus_failed_list  = [2, 3, 5, 7, 9, 13, 16, 17, 19, 27, 29, 30, 31, 32, 40, 59, 60, 61, 64, 67, 68, 69, 70, 72, 73, 74, 77, 78, 79, 81, 82, 90, 91, 92, 101, 102, 103, 104, 108, 110, 112, 120, 122, 124, 125, 127, 130, 132, 138, 140, 142, 145, 147, 148, 151, 154, 155, 165, 169, 171, 172, 174, 175, 181, 187, 188, 189, 193, 197, 198, 200, 201, 203, 207, 210, 212, 214, 215, 216, 218, 219, 220, 222, 223, 226, 229, 230, 234, 236, 238, 239, 240, 243, 245, 247, 250, 254, 255, 257, 258, 259, 267, 270, 279, 280, 282, 283, 288, 290, 292, 293, 294, 297, 300, 301, 302, 309, 313, 318, 320, 322, 332, 333, 334, 338, 339, 342, 344, 346, 348, 349, 351, 354, 357, 358, 359, 361, 365, 368, 369, 370, 371, 378, 379, 380, 385, 387, 389, 390, 392, 393, 394, 395, 399, 409, 416, 418, 419, 423, 430, 435, 436, 441, 443, 444, 445, 446, 447, 448, 452, 454, 455, 459, 460, 461, 462, 463, 464, 465, 467, 468, 469, 470, 471, 472, 473, 478, 480, 484, 487, 488, 491, 493, 495, 496]
        for sub_folder in ["annotations", "images", "labels"]:
            sub_dir = os.path.join(self.dataset_path, "Transportation-train", sub_folder)
            if os.path.isdir(sub_dir):
                # 列出該目錄中的所有檔案
                all_files = os.listdir(sub_dir)
                for i in bus_failed_list:
                    if i < 10:
                        target_file = f"Bus_0{i}.txt"
                    else:
                        target_file = f"Bus_{i}.txt"
                    if sub_folder == 'images':
                        target_file = target_file.replace(".txt", ".jpg")

                    if target_file in all_files:
                        target_file_location = f'{self.model_dataset_path}/{sub_folder}/{target_file}'
                        if os.path.exists(target_file_location):
                            os.remove(target_file_location)
                            print(f"{target_file_location} 已刪除。")


            # for i in range (1, 500):
            #     if i < 10 :
            #
            #
            #     else:


# if __name__ == '__main__':
#     test_ = DataPreProcessing(kind_list=['Bus', 'Car'])
#     test_.combine_and_split()