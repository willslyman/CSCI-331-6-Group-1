from ultralytics import YOLO
import cv2
import os
import random

if __name__ == '__main__':
    model = YOLO('runs/detect/exp_medium_dataset/weights/best.pt')

    imgs = os.listdir('data/car/test/images')

    for x in range(20):
        path = imgs.pop(random.randint(0, len(imgs)-1))
        
        results = model.predict(os.path.join('data/car/test/images', path))

        for r in results:
            im = r.plot()
            cv2.imshow("Prediction", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
