# Interpretable Bioacoustic Classifiers
A training and evaluation pipeline for analysing Passive Acoustic Monitoring audio. Train an intra-frame shift invariant Variational Autoencoder and linear species classifiers using this pipeline. View our project page [here](https://m4gpi.github.io/interpretable_bioacoustic_classifiers/).

## Usage
This pipeline uses Hydra for instantiating objects using configuration files for easy persistence and configurability of experiments. There is a small learning curve to using Hydra. The `config` directory contains object instantiation configuration files. These reference specific classes in the code.

```bash
# install dependencies
uv sync

# train a SIVAE with default settings
uv run main.py train +experiment=nifti_vae data=your_data_module paths.results_dir=/path/to/results

# train a species detector with default settings
uv run main.py train +experiment=species_detector data.root=/path/to/embeddings+labels paths.results_dir=/path/to/results

# evaluate BirdNET on your dataset
uv run main.py eval +experiment=birdnet data=your_data_module paths.results_dir=/path/to/results
```

A hyper-parameter sweep for species detectors can be done using hydra and joblib, in this case, the experiment is run 7 times independently using K-folds cross-validation where each run is an independent fold.

```bash
uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=7 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  data.num_folds=7 \
  data.fold_id=0,1,2,3,4,5,6 \
  model.attn_learning_rate=1e-2,5e-3 \
  model.clf_learning_rate=1e-1,5e-2 \
```
