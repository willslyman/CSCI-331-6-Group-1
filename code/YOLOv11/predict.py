from ultralytics import YOLO
import cv2

model = YOLO('PUT YOUR PATH TO ONNX FILE HERE')

results = model.predict('PUT PATH TO WANT TO PREDICT IMAGE HERE', imgsz=640)

for r in results:
    im = r.plot()
    cv2.imshow("Prediction", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()