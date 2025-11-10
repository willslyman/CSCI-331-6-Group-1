from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')

    results = model.train(
        data='conf.yaml',
        epochs=200,
        patience=25,
        batch=16,
        imgsz=640,
        workers=8,
        name='exp_medium_dataset'
    )

    model.val()
    model.export(format='onnx')
