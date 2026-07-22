# Interpretable Bioacoustic Classifiers

A training and evaluation pipeline for analysing Passive Acoustic Monitoring audio. Train an intra-frame shift invariant Variational Autoencoder and multi-label multiple-instance logistic regression model(s) using this pipeline. 

View our project page [here](https://m4gpi.github.io/interpretable_bioacoustic_classifiers/).

## Usage
This pipeline uses hydra for instantiating objects using configuration files for easy persistence and configurability of experiments. There is a small learning curve to using Hydra. The `config` directory contains object instantiation configuration files. These reference specific classes in the code.

```bash
# install dependencies
uv sync

# train a VAE with default settings
uv run main.py train +experiment=vae data=your_data_module paths.results_dir=/path/to/results

# train a SIVAE with default settings
uv run main.py train +experiment=sivae data=your_data_module paths.results_dir=/path/to/results

# embed vae / sivae features for second stage
uv run main.py eval +experiment=sivae data=your_data_module evaluator=lightning_predict ckpt_path=/path/to/model.ckpt paths.results_dir=/path/to/embeddings

# train a species detector with default settings
uv run main.py train +experiment=mil_species_detector data.root=/path/to/embeddings paths.results_dir=/path/to/results

# evaluate BirdNET on your dataset
uv run main.py eval +experiment=birdnet_predictions data=your_data_module paths.results_dir=/path/to/results data.num_workers=16

# embed your dataset using BirdNET
uv run main.py eval +experiment=birdnet_embeddings data=your_data_module paths.results_dir=/path/to/embeddings data.num_workers=16

# train a species detector on birdnet embeddings
uv run main.py train +experiment=birdnet_mil_species_detector data.root=/path/to/embeddings paths.results_dir=/path/to/results
```

A hyper-parameter sweep for species detectors can be done using hydra and joblib, in this case, the experiment is run 5 times independently using K-folds cross-validation where each run is an independent fold.

```bash
uv run main.py train +experiment=mil_species_detector_sweep scope=SO_UK

uv run main.py train +experiment=birdnet_mil_species_detector_sweep scope=SO_UK
```
