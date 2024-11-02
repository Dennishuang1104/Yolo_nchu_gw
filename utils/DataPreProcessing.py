from PIL import Image
from tqdm import tqdm
from yolov3_pytorch.utils.general import os, Path
import glob


class DataPreProcessing:
    def __init__(self, kind_list: list):
        self.kind_list = kind_list
        self.img_type = 'jpg'
        self.dataset_path = 'datasets/'
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
        # Convert VisDrone box to YOLO xywh box
        # 將標記框框轉換成 yolo 對應的xywh
        dw = 1. / size[0]  # 用1 當作所有
        dh = 1. / size[1]
        box_tuple = (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh
        return box_tuple

    def visdrone2yolo(self, kind_dir):
        full_kind_dir = Path(self.dataset_path + kind_dir)
        print(full_kind_dir)
        (full_kind_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
        # pbar = tqdm((full_kind_dir / 'annotations').glob('*.txt'), desc=f'Converting {kind_dir}')
        pbar = tqdm(
            [f for f in (full_kind_dir / 'annotations').glob(f'{kind_dir}_*.txt')
             if int(f.stem.split('_')[1]) < 100],
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



# if __name__ == '__main__':
#     test_ = DataPreProcessing(kind_list=['Bus', 'Car'])
#     test_.visdrone2yolo(dataset_dir='Car')
