from ultralytics import YOLO

model = YOLO('yolo11n.yaml')

results = model.train(data='conf.yaml', epochs=1)
results = model.val()

results = model.export(format='onnx')