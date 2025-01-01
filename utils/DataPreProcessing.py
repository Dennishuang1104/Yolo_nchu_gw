from PIL import Image
from tqdm import tqdm
from yolov3_pytorch_modules.utils.general import os, Path
import shutil
import glob


class DataPreProcessing:
    def __init__(self, kind_list: list):
        self.kind_list = kind_list
        self.counts_data = 550  # 拿幾張訓練
        self.img_type = 'jpg'
        self.file_type_list = ['images', 'labels']
        self.dataset_path = 'datasets/'
        self.diy_dataset_path = 'datasets/LabelsDIY'
        # self.model_dataset_path = 'datasets/Transportation-train'
        self.model_dataset_path = 'datasets/Jaychou-train'
        # self.jaychou_dataset_path = 'datasets/Jaychou'
        self.jaychou_dataset_path = 'datasets/FakeChou'
        self.class_dict = {
            "Jaychou": 0,
            "HumanFace": 1,
            "FakeChou": 2
        }
        # self.class_dict = {
        #     "Car": 0,
        #     "Bus": 1,
        #     "Bicycle": 2,
        #     "Motorcycle": 3
        # }

    def separate_chou_from_txt(self):
        """
        將標記好的資料進行切割成各檔案
        :return:
        """
        input_file = "datasets/Jaychou-raw-data/train/_annotations.txt"  # 原始檔案名稱

        with open(input_file, "r") as file:
            lines = file.readlines()

        # 遍歷每一行並處理內容
        for line in lines:
            parts = line.strip().split(" ")  # 分隔檔名與數字
            raw_filename = parts[0]  # 取得原始檔案名稱

            # 擷取 `jay_chou_XXX` 部分作為輸出檔名
            filename_parts = raw_filename.split("_")

            serial_number = filename_parts[2]  # 擷取序號部分
            output_filename = f"./jay_chou_{serial_number}.txt"

            # 移除檔名部分，保留數字
            data = " ".join(parts[1:])  # 合併數字部分

            # 建立對應的 .txt 檔案
            with open(output_filename, "w") as outfile:
                outfile.write(data)  # 將數字寫入新檔案

    def rename_chou_images_name(self):
        """
        transform images jaychou images names
        """
        # 遍歷 Images 資料夾中的所有檔案
        images_dir = self.jaychou_dataset_path + '/images'
        for filename in os.listdir(images_dir):
            # 檢查是否為檔案
            if os.path.isfile(os.path.join(images_dir, filename)):
                # 找到檔名前半部分（移除 ".rf.xxx.jpg" 部分
                new_name = filename.replace('fake_jay_chou', 'fake_jaychou')
                new_name = "_".join(new_name.split("_")[:3]).replace('jpg.jpg', 'jpg')
                # new_name = new_name + '.jpg'
                # print(new_name)
                # new_name = new_name.replace('fake_jay_chou', 'fake_jaychou')
                #
                # # 拼接完整路徑
                old_path = os.path.join(images_dir, filename)
                new_path = os.path.join(images_dir, new_name)
                #
                # # 重新命名檔案
                os.rename(old_path, new_path)
                print(f"已將檔案重命名：{filename} -> {new_name}")

        txt_dir = self.jaychou_dataset_path + '/annotations'
        for filename in os.listdir(txt_dir):
            # 檢查是否為檔案
            if os.path.isfile(os.path.join(txt_dir, filename)):
                # 找到檔名前半部分（移除 ".rf.xxx.jpg" 部分）
                new_name = filename.replace('fake_jay_chou', 'fake_jaychou')
                new_name = "_".join(new_name.split("_")[:3]).replace('txt.txt', 'txt')
                new_name = new_name + ".txt"

                # 拼接完整路徑
                old_path = os.path.join(txt_dir, filename)
                new_path = os.path.join(txt_dir, new_name)

                # 重新命名檔案
                os.rename(old_path, new_path)
                print(f"已將檔案重命名：{filename} -> {new_name}")

        print("所有檔案已完成重命名！")

    def get_filenames(self, kind):
        # step 1 把Label 內.txt資料搬出來
        txt_dir_path = f'{self.dataset_path}/{kind}/annotations'
        jpg_dir_path = f'{self.dataset_path}/{kind}/images'
        if not os.path.exists(jpg_dir_path):
            os.makedirs(jpg_dir_path)
        if not os.path.exists(txt_dir_path):
            os.makedirs(txt_dir_path)

        # step 2 把對應ID 取代成
        id_list = []
        directory = os.path.join(os.path.dirname(__file__), jpg_dir_path).replace('utils/', '')
        for idx, filename in enumerate(os.listdir(directory), start=1):
            if filename.endswith(f'.{self.img_type}'):
                id_list.append(filename.replace(f'.{self.img_type}', ''))

        # step 3 將ID list 元素存成新的/.txt, .jpg
        for idx, id_ in enumerate(id_list, start=1):
            original_jpg_name = f'{id_}.jpg'
            original_txt_name = f'{id_}.txt'
            jpg_filename = f"{kind}_{str(idx).zfill(2)}.jpg"
            txt_filename = f"{kind}_{str(idx).zfill(2)}.txt"

            original_txt_path = os.path.join(txt_dir_path, original_txt_name)
            new_txt_path = os.path.join(txt_dir_path, txt_filename)
            os.rename(original_txt_path, new_txt_path)

            original_jpg_path = os.path.join(jpg_dir_path, original_jpg_name)
            new_jpg_path = os.path.join(jpg_dir_path, jpg_filename)
            os.rename(original_jpg_path, new_jpg_path)

    def rename_kind(self, dir_name):
        txt_dir_path = f'{self.dataset_path}/{dir_name}/labels'
        jpg_dir_path = f'{self.dataset_path}/{dir_name}/images'
        if os.path.exists(jpg_dir_path):
            dir_path = f'../datasets/{dir_name}'
            txt_directory = os.path.join(os.path.dirname(__file__), txt_dir_path)
            for filename in os.listdir(txt_directory):
                original_path = os.path.join(txt_dir_path, filename)
                new_path = os.path.join(dir_path, filename)
                os.rename(original_path, new_path)

    def transfer_to_labelsdiy(self):
        """
        將手動標注好的檔案統一轉到LabelsDIY 裡面，並且命名規則為Name_ID 升冪排列
        """
        destination_dir = f"{self.dataset_path}/LabelsDIY/"
        txt_dir_path = f'{self.dataset_path}/LabelsDIY/labels'
        jpg_dir_path = f'{self.dataset_path}/LabelsDIY/images'
        for t in self.file_type_list:
            destination_type_dir = destination_dir + t
            if not os.path.exists(destination_type_dir):
                os.makedirs(destination_type_dir)
            else:
                shutil.rmtree(destination_type_dir)
                os.makedirs(destination_type_dir, exist_ok=True)

        # 依照對應的目錄做資料處理
        for name in self.diy_list:
            image_dir = destination_dir.replace('LabelsDIY/', f'{name}/images')
            label_dir = destination_dir.replace('LabelsDIY/', f'{name}/labels')
            file_list = []
            # 防呆機制避免比數不同 造成錯誤
            for file_name in os.listdir(image_dir):
                filename = file_name.replace(".jpg", "")
                file_list.append(filename)

            for idx, id_ in enumerate(file_list, start=1):
                original_jpg_name = f'{id_}.jpg'
                original_txt_name = f'{id_}.txt'
                jpg_filename = f"{name}_{str(idx).zfill(2)}.jpg"
                txt_filename = f"{name}_{str(idx).zfill(2)}.txt"

                original_txt_path = os.path.join(label_dir, original_txt_name)
                new_txt_path = os.path.join(txt_dir_path, txt_filename)
                shutil.copy(original_txt_path, new_txt_path)

                original_jpg_path = os.path.join(image_dir, original_jpg_name)
                new_jpg_path = os.path.join(jpg_dir_path, jpg_filename)
                shutil.copy(original_jpg_path, new_jpg_path)

    def process_label_file(self, input_path, output_path):
        """
        處理單一標記文件，將多邊形座標轉為 Bounding Box 格式。
        """
        with open(input_path, 'r') as infile:
            lines = infile.readlines()

        modified_lines = []
        for line in lines:
            elements = line.split()
            if len(elements) < 2:
                continue  # 跳過空行或格式不正確的行

            class_id = elements[0]
            points = list(map(float, elements[1:]))

            # 提取多邊形的 (x, y) 座標
            x_coords = points[0::2]  # 偶數索引是 x 座標
            y_coords = points[1::2]  # 奇數索引是 y 座標

            # 計算 Bounding Box
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # 生成新的標記格式 , 並訂正標記錯誤的資訊
            if int(class_id) == 2:
                class_id = "0"
            elif int(class_id) == 0:
                class_id = "3"
            bounding_box = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            modified_lines.append(bounding_box)

            # 將修改後的內容寫入輸出文件
            with open(output_path, 'w') as outfile:
                outfile.write("\n".join(modified_lines) + "\n")

    def tf_annotation_to_bbox_batch(self):
        labels_dir = f"{self.dataset_path}/Lara/labels"
        labels_copy_dir = f"{self.dataset_path}/Lara/labels_copy"

        labels_exists = os.path.exists(labels_dir)
        labels_copy_exists = os.path.exists(labels_copy_dir)

        # 重置資料夾
        if labels_exists and labels_copy_exists:
            print("`labels` 和 `labels_copy` 目錄均存在，無需執行動作。")
        elif labels_exists and not labels_copy_exists:
            shutil.copytree(labels_dir, labels_copy_dir)
            shutil.rmtree(labels_dir)
            os.makedirs(labels_dir, exist_ok=True)
            print(f"已清空並重新建立 `{labels_dir}` 資料夾。")
        else:
            print(f"目錄 `{labels_dir}` 不存在，請確認資料夾配置。")

        # 遍歷所有 `.txt` 文件
        for filename in os.listdir(labels_copy_dir):
            if filename.endswith('.txt'):
                source_path = os.path.join(labels_copy_dir, filename)
                target_path = os.path.join(labels_dir, filename)
                self.process_label_file(source_path, target_path)
                print(f"已處理文件：{source_path} -> {target_path}")

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
                    if row[0] == "0":
                        cls = "2"
                    else:
                        cls = "0"
                    box = tuple(map(float, row[1:5]))
                    lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                    with open(str(img_file).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                        fl.writelines(lines)  # write label.txt

    def clean_files_name(self):
        # 設定標記檔案的資料夾路徑
        annotations_folder = "datasets/Human_face/annotations"
        # 遍歷資料夾中的所有檔案
        for filename in os.listdir(annotations_folder):
            # 確保只處理標記檔案，例如 .txt 檔案
            if filename.endswith(".txt"):
                file_path = os.path.join(annotations_folder, filename)

                # 讀取檔案內容
                with open(file_path, "r") as file:
                    lines = file.readlines()

                # 處理每行內容
                updated_lines = []
                for line in lines:
                    # 將 "Human face" 替換為 "Human_face"
                    updated_line = line.replace("Human_face", "HumanFace")
                    updated_lines.append(updated_line)

                # 將修改後的內容寫回原檔案
                with open(file_path, "w") as file:
                    file.writelines(updated_lines)

        print("所有檔案已成功處理！")

    def relocated_file_name(self):
        directory = "datasets/Human_face/images"

        # 讀取目錄中的所有 .txt 檔案
        txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

        # 排序檔案名稱（按字母或數字順序）
        txt_files.sort()

        # 依序重命名
        for index, file in enumerate(txt_files):
            # 新的檔案名稱，格式為 Human_face_XX.txt（補零到兩位數）
            new_name = f"Human_face_{index:02d}.jpg"

            # 獲取完整路徑
            old_path = os.path.join(directory, file)
            new_path = os.path.join(directory, new_name)

            # 重命名檔案
            os.rename(old_path, new_path)

        print("檔案已重新命名完成！")

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
        # 把自標檔案複製到模型訓練目錄
        source_dir = self.diy_dataset_path
        target_dir = self.model_dataset_path

        # for t in self.file_type_list:
        #     full_source_dir = source_dir + "/" + t
        #     full_target_dir = target_dir + "/" + t
        #
        #     for filename in os.listdir(full_source_dir):
        #         source_path = os.path.join(full_source_dir, filename)
        #         target_path = os.path.join(full_target_dir, filename)
        #         shutil.copy(source_path, target_path)




# if __name__ == '__main__':
#     test_ = DataPreProcessing(kind_list=['Bus', 'Car'])
#     test_.combine_and_split()