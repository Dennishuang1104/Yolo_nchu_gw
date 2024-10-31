import os
import glob


def get_filenames(kind, data_type='jpg'):
    # step 1 把Label 內.txt資料搬出來
    txt_dir_path = f'../datasets/{kind}/Label'
    jpg_dir_path = f'../datasets/{kind}/images'
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
        if filename.endswith(f'.{data_type}'):
            id_list.append(filename.replace(f'.{data_type}', ''))

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

    print(f'{kind} work done!')


if __name__ == '__main__':
    # get_files(data_type='jpg')
    jpg_name = get_filenames(kind='Bus')
    # print(jpg_name)
    # print(len(jpg_name))
    # / Users / dennis_huang / Github_project / Yolo_nchu_gw / utils /../ datasets / Bus / Label / 20225397f642ff82.txt