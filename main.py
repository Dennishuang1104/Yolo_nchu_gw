from utils.DataPreProcessing import DataPreProcessing
import subprocess


def run_training():
    # 定義要執行的命令以及所需的參數
    command = [
        "python", "yolov3_pytorch_modules/train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", "1",
        "--weights", "",
        "--cfg", "yolov3_pytorch_modules/models/yolov3.yaml",
        "--hyp", "yolov3_pytorch_modules/data/hyps/hyp.scratch-high.yaml",
        "--data", "yolov3_pytorch_modules/data/Transportation.yaml",
        "--cache"
    ]

    # 使用 subprocess.run 執行命令
    result = subprocess.run(command, capture_output=True, text=True)

    # 打印訓練的輸出訊息
    print("訓練過程輸出:")
    print(result.stdout)

    # 錯誤處理
    if result.returncode != 0:
        print("執行出現錯誤:")
        print(result.stderr)



def main():
    # yolo_tackler = DataPreProcessing(kind_list=['Car', 'Bus'])
    # yolo_tackler.visdrone2yolo('Bus')
    # yolo_tackler.visdrone2yolo('Car')
    # yolo_tackler.combine_and_split()
    # yolo_tackler.clean_dirty_data()
    run_training()


if __name__ == '__main__':
    main()