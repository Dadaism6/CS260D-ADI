# CS260D-ADI
This is the repo for UCLA CS260D Proejct.

## Using CREST
To install the required packages, run `pip install -r requirements.txt`.

To use, set `args.smtk=0` in `utils/argument.py`, run `python train.py`.

To load our dataset, add the dataset loading code in `datasets/indexed_data.py`.

To load our model, create new file in `/models` folder, which contains the model class. Then add the model name of the choices of `--arch` argument in `utils/argument.py`. Modify the code in `crest_train.py` in Line 79.

Note that in `crest_trainer.py`, we have `data, target, data_idx = batch = next(self.train_iter)` in Line 57, 65. Make sure the dataset aligns with this format.

## Using CRAIG

To use, set `args.smtk=1` in `utils/argument.py`, run `python train.py`.