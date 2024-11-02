from utils.DataPreProcessing import DataPreProcessing


def main():
    yolo_tackler = DataPreProcessing(kind_list=['Car', 'Bus'])
    yolo_tackler.visdrone2yolo('Bus')


if __name__ == '__main__':
    main()