from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    results = model.train(data="./datasets/dataset.yaml", epochs=1000, imgsz=(288, 512), device=0, batch=16)
