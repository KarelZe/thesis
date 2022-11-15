## Save and Load Data and Models by Hash
- https://docs.wandb.ai/guides/artifacts/track-external-files

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact('mnist', type='dataset')

artifact.add_reference('s3://my-bucket/datasets/mnist')

# Track the artifact and mark it as an input to

# this run in one swoop. A new artifact version

# is only logged if the files in the bucket changed.

run.use_artifact(artifact)

artifact_dir = artifact.download()

# Perform training here...
```

```python
import boto3

import wandb

run = wandb.init()

# Training here...

s3_client = boto3.client('s3')

s3_client.upload_file('my_model.h5', 'my-bucket', 'models/cnn/my_model.h5')

model_artifact = wandb.Artifact('cnn', type='model')

model_artifact.add_reference('s3://my-bucket/models/cnn/')

run.log_artifact(model_artifact)
```

## Optuna Conditional Search Spaces
- See e. g., https://github.com/optuna/optuna/issues/1809
```python
classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
if classifier_name == "SVC":
    svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
else:
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
    classifier_obj = sklearn.ensemble.RandomForestClassifier(
    max_depth=rf_max_depth, n_estimators=10
    )
```

## Optuna an Weights and Biases Integration
- https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.WeightsAndBiasesCallback.html
```python
wandb_kwargs = {"project": "optuna-wandb-example"}

wandbc = WeightsAndBiasesCallback(metric_name="accuracy", wandb_kwargs=wandb_kwargs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, callbacks=[wandbc])
```

## Logging custom metrics
```python
@wandbc.track_in_wandb()
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    wandb.log({"power": 2, "base of metric": x - 2})

    return (x - 2) ** 2
```

## Saving and Pruning Callback
```python
# found at https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study/62164601#62164601

import lightgbm as lgb

import numpy as np

import optuna

import sklearn.datasets

import sklearn.metrics

from sklearn.model_selection import train_test_split

  
  

class Objective:

  

    def __init__(self):

        self.best_booster = None

        self._booster = None

  

    def __call__(self, trial):

        data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)

        train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

        dtrain = lgb.Dataset(train_x, label=train_y)

        dvalid = lgb.Dataset(valid_x, label=valid_y)

  

        param = {

            "objective": "binary",

            "metric": "auc",

            "verbosity": -1,

            "boosting_type": "gbdt",

            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),

            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),

            "num_leaves": trial.suggest_int("num_leaves", 2, 256),

            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),

            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),

            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),

            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

        }

  

        # Add a callback for pruning.

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

        gbm = lgb.train(

            param, dtrain, valid_sets=[dvalid], verbose_eval=False, callbacks=[pruning_callback]

        )

  

        self._booster = gbm

  

        preds = gbm.predict(valid_x)

        pred_labels = np.rint(preds)

        accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)

        return accuracy

  

    def callback(self, study, trial):

        if study.best_trial == trial:

            self.best_booster = self._booster
		# do something with it...
```

## Out-of-memory callback

```python
# found at: https://optuna.readthedocs.io/en/latest/faq.html#how-to-save-machine-learning-models-trained-in-objective-functions
def objective(trial):
    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_int("y", -5, 5)
    return x + y

study = optuna.create_study()
study.optimize(objective, n_trials=10, gc_after_trial=True)

# `gc_after_trial=True` is more or less identical to the following.
study.optimize(objective, n_trials=10, callbacks=[lambda study, trial: gc.collect()])
```

### Observe unfinished trials
- https://cloud.google.com/sql
- https://docs.sqlalchemy.org/en/20/dialects/mysql.html#using-mysqldb-with-google-cloud-sql
- https://stackoverflow.com/questions/10763171/can-sqlalchemy-be-used-with-google-cloud-sql (expensive; use with free credits 
  $ 300?)
- https://www.cdata.com/kb/tech/dynamodb-python-sqlalchemy.rst (dynamo db?)

```python
# https://optuna.readthedocs.io/en/latest/faq.html#how-to-save-machine-learning-models-trained-in-objective-functions
import optuna
from optuna.storages import RetryFailedTrialCallback

storage = optuna.storages.RDBStorage(
    url="sqlite:///:memory:",
    heartbeat_interval=60,
    grace_period=120,
    failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
)

study = optuna.create_study(storage=storage)
```

### Save and Resume Optuna Studies

```python
# non dbms
# https://optuna.readthedocs.io/en/latest/faq.html#how-can-i-save-and-resume-studies
study = optuna.create_study()
joblib.dump(study, "study.pkl")

study = joblib.load("study.pkl")
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

#dbms solution
# https://optuna.readthedocs.io/en/latest/tutorial/20_recipes/001_rdb.html#rdb
```

## Save CatBoost Models

- https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
- Only `cbm` supports categorical data.
```python
save_model(fname, format="cbm", export_parameters=None, pool=None)
```

## Optuna CLI
- see https://optuna.readthedocs.io/en/stable/reference/cli.html
- see https://neptune.ai/blog/optuna-guide-how-to-monitor-hyper-parameter-optimization-runs

```python
def objective(trial): x = trial.suggest_uniform('x', -10, 10) return (x - 2) ** 2
```

```bash
$ STUDY_NAME=`optuna create-study --storage sqlite:///example.db` $ optuna study optimize foo.py objective --n-trials=100 --storage sqlite:///example.db --study-name $STUDY_NAME
```

## Save and Load PyTorch Models at Checkpoint
- found at https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
### Save
```python
# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

[torch.save](https://pytorch.org/docs/stable/generated/torch.save.html#torch.save "torch.save")({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
```
### Load
```python
model = Net()
optimizer = [optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD "torch.optim.SGD")(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load "torch.load")(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

