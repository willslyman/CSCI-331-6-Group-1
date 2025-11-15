from ultralytics import YOLO
from ray import tune

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')

    # Define search space
    # Search for the best values for these hyperparams
    # between respective ranges
    search_space = {
        "lr0": tune.uniform(1e-5, 1e-1),
        "batch": tune.randint(4, 16),
        "epochs": tune.randint(50, 200)
    }

    results = model.tune(
        data="C:/Users/12162/OneDrive/Documents/RIT Files/IntroToAI/CSCI-331-6-Group-1/code/YOLOv11/conf.yaml",
        space=search_space,
        patience=25, # Not a standard hyperparameter
        imgsz=640,
        workers=4,
        name='hyperparameter_tuning',
        use_ray=True,
        device=0,
        gpu_per_trial=1,
        resume=True
    )

    # Add resume=True to the tune command if resuming already started training
