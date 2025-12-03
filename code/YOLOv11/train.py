from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')

    results = model.train(
        data="./code/YOLOv11/conf.yaml",
        epochs=95, # Best Hyperparam from Tuning
        patience=25,
        batch=10, # Best Hyperparam from Tuning
        imgsz=640,
        workers=8,
        lr0=0.07412, # Best Hyperparam From Tuning
        name='exp_medium_dataset'
    )

    model.val()
    model.export(format='onnx')
