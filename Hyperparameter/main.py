import os
import optuna
from optuna.storages import RetryFailedTrialCallback

from Optuna_template_modification import objective


if __name__ == "__main__":
    # define the storage
    storage = optuna.storages.RDBStorage(
        "sqlite:///example.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    # create an experiment and determine the direction='maximize' or 'minimize'
    study = optuna.create_study(
        storage=storage, study_name="pytorch_checkpoint", direction="maximize", load_if_exists=True
    )
    # define objective here
    study.optimize(objective, n_trials=50, timeout=600)

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print(" Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
