# CSCI-331-6-Group-1

## Abstract: 
This is our group project for Intro to Artificial Intelligence. It uses YOLO v11 to identify traffic signs in images. The project includes both a training run (`train.py`), tuning run (`tune.py`), and a prediction script for testing the model (`predict-random.py`).

## List of Developers
- Adam Rigdon
- Leo Grover
- Will Slyman

## How to Run
- **WARNING:** Depending on your OS and your Python install, you may need to tweaks the directory paths in the code to be **absolute** to make it run.
- Import all dependcies required on Python version 13.12.x.
- To run the training, run `train.py`.
- To find the best Hyperparameters, open `tune.py` and set `RUN_TUNER` to `True`, and then run the file.
- To get graphs for the best Hyperparameters after tuning, en `tune.py` and set `RUN_TUNER` to `False`, and then run the file.
- To test the model on 20 random images, run `predict-random.py`.