from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')

    # Define search space
    search_space = {
        "lr0": (1e-5, 1e-1),
        "batch": [int(4), int(8), int(16)],
        "epochs": [int(50), int(100), int(200)]
    }

    results = model.tune(
        data='D:\Profiles\whs1585\Desktop\AIProject\code\YOLOv11\conf.yaml',
        space=search_space,
        patience=25, # Not a standard hyperparameter
        imgsz=640,
        workers=16,
        name='hyperparameter_tuning'
    )

    # Add Resume=True to the tune command if resuming already started training

    model.export(format='onnx')
