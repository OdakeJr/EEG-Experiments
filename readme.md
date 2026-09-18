# EEG Experiments

Research code for EEG machine-learning experiments, with focus on EEG representations and generalization across subjects and sessions.

## Setup

Create and activate a Conda environment:

```bash
conda create -n eeg311 python=3.11 -y
conda activate eeg311
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project requirements:

```bash
python -m pip install -r requirements.txt
```

Run grid experiments:

```bash
chmod +x run_feature_grid.sh
./run_feature_grid.sh
```