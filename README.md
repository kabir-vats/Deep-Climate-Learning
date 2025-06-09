# Deep-Climate-Learning

This repository contains code and models used to compete in the competition: [Deep Learning for Climate Emulation](https://www.kaggle.com/competitions/cse151b-spring2025-competition/overview)

Our team HK(N+1) achieved second place, with a private leaderboard score of 0.65.

Repository organization:

```
(●'◡'●)
├─ .gitignore
├─ README.md
├─ climate_prediction
│  ├─ __init__.py
│  ├─ data_loader.py              // data module (both regular and decadal)
│  ├─ loss.py                     // custom loss
│  ├─ models
│  │  ├─ CNN_LSTM.py
│  │  ├─ CNN_MLP.py
│  │  ├─ ClimaX_Transformer.py
│  │  └─ ConvLSTM.py
│  ├─ train.py                    // lightning module
│  └─ util.py                     // utilities, visualizations, etc.
├─ notebooks
│  ├─ PR-CNN-MLP.ipynb
│  └─ TAS_ClimaX.ipynb
└─ requirements.txt
```


