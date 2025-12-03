from ultralytics import YOLO
from ray import tune
from ray.tune import ExperimentAnalysis
import sys
import matplotlib.pyplot as plt
from pathlib import Path

RUN_TUNER = False

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')

    if not RUN_TUNER:
        experiment_path = Path("./code/YOLOv11/runs/detect/hyperparameter_tuning").resolve()
        analysis = ExperimentAnalysis(str(experiment_path))
        
        df = analysis.dataframe()
        
        ### Print best hyperparams for mAP50 ###
        # From https://docs.ultralytics.com/guides/yolo-performance-metrics/#class-wise-metrics
        # mAP50: Mean average precision calculated at an intersection over union (IoU) threshold of 0.50. 
        
        # It's a measure of the model's accuracy considering only the "easy" detections.
        print("Best Config (mAP50): ", analysis.get_best_config(
            metric="metrics/mAP50(B)",
            mode="max"
        ))

        ### Print best hyperparams for mAP50-95 ###
        # From https://docs.ultralytics.com/guides/yolo-performance-metrics/#class-wise-metrics
        # The average of the mean average precision calculated at varying IoU thresholds,
        # ranging from 0.50 to 0.95.

        # It gives a comprehensive view of the model's performance across
        # different levelsof detection difficulty.
        print("Best Config (mAP50-95): ", analysis.get_best_config(
            metric="metrics/mAP50-95(B)",
            mode="max"
        ))
        
        fig1, ax1 = plt.subplots() # mAP50-95 over trials
        ax1.set_title('mAP50-95(B) v.s. Trials')
        ax1.set_xlabel('Trials')
        ax1.set_ylabel('mAP50-95(B)')
       
        fig2, ax2 = plt.subplots() # F1 Scores over trials
        ax2.set_title('F1 Scores v.s. Trials')
        ax2.set_xlabel('Trials')
        ax2.set_ylabel('F1 Score')

        fig3, ax3 = plt.subplots() # mAP50-95 v.s. Epochs (for all iterations)
        ax3.set_title('Overall Hyperparameter Tuning Results:\nmAP50-95(B) v.s. Epochs')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('mAP50-95(B)')

        fig4, ax4 = plt.subplots() # Precision over trials
        ax4.set_title('Precision v.s. Trials')
        ax4.set_xlabel('Trials')
        ax4.set_ylabel('Precision')

        fig5, ax5 = plt.subplots() # Recall over trials
        ax5.set_title('Recall v.s. Trials')
        ax5.set_xlabel('Trials')
        ax5.set_ylabel('Recall')
        
        # Ensure only one value per trial
        plot_df = df.drop_duplicates(subset='trial_id', keep='first').sort_values(by='trial_id')
        
        # Get trials, sorted from 0 to 9
        x_trials = plot_df['trial_id'].rank(method='dense').astype(int) 
        ax1.set_xticks(x_trials)
        ax2.set_xticks(x_trials)
        ax4.set_xticks(x_trials)
        ax5.set_xticks(x_trials)

        # Get accuracy and f1 score
        y_accuracy = plot_df["metrics/mAP50-95(B)"]
        y_f1 = (2 * plot_df["metrics/precision(B)"] * plot_df["metrics/recall(B)"]) / (plot_df["metrics/precision(B)"] + plot_df["metrics/recall(B)"])
        
        y_precision = plot_df["metrics/precision(B)"]
        y_recall = plot_df["metrics/recall(B)"]
        
        # Plot change in all over 10 mutations
        ax1.plot(x_trials, y_accuracy)
        ax2.plot(x_trials, y_f1)
        ax4.plot(x_trials, y_precision)
        ax5.plot(x_trials, y_recall)

        # Plot combination of all iterations
        dfs = dict(sorted(analysis.trial_dataframes.items()))
        for path, trial_df in dfs.items():
            lr0 = str(round(trial_df['config/lr0'][0], 3))
            batch = str(trial_df['config/batch'][0])
            id = path[-1]

            ax3.plot(
                trial_df['training_iteration'],
                trial_df['metrics/mAP50-95(B)'],
                alpha=0.5,
                label=(id + ": lr0 = " + lr0 + ", batch = " + batch)
            )
        ax3.legend()
        plt.show()

        # Exit
        sys.exit(0)

    # Define search space
    # Search for the best values for these hyperparams
    # between respective ranges
    search_space = {
        "lr0": tune.uniform(1e-5, 1e-1),
        "batch": tune.randint(4, 16),
        "epochs": tune.randint(50, 200)
    }
    
    while True:
        try:
            results = model.tune(
                data=Path("./code/YOLOv11/conf.yaml").resolve(),
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
            if results != None:
                break
        except:
            pass

    # Add resume=True to the tune command if resuming already started training
