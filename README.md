# YAIB-models
Models trained for the publication of [Yet Another ICU Benchmark](https://github.com/rvandewater/YAIB). The models are named in the following manner: _dataset_task_model_cv-repetition_cv-fold_. Please not that it is possible that the performance of the classification models might deviate slightly from the official paper results due to major improvements to YAIB in the meantime. We hope to confirm the results once YAIB is out of alpha.
## Datasets
| Dataset                 | [MIMIC-III](https://physionet.org/content/mimiciii/) / [IV](https://physionet.org/content/mimiciv/) | [eICU-CRD](https://physionet.org/content/eicu-crd/) | [HiRID](https://physionet.org/content/hirid/1.1.1/) | [AUMCdb](https://doi.org/10.17026/dans-22u-f8vd) |
|-------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|--------------------------------------------------|
| Admissions              | 40k / 73k                                                                                           | 200k                                                | 33k                                                 | 23k                                              |
| Frequency (time-series) | 1 hour                                                                                              | 5 minutes                                           | 2 / 5 minutes                                       | up to 1 minute                                   |
| Origin                  | USA                                                                                                 | USA                                                 | Switzerland                                         | Netherlands                                      |

## Tasks
| No  | Task Theme                | Frequency        | Type                                | 
|-----|---------------------------|--------------------|-------------------------------------|
| 1   | ICU Mortality             | Once per Stay (after 24H) | Binary Classification  |
| 2   | Acute Kidney Injury (AKI) | Hourly (within 6H) | Binary Classification |
| 3   | Sepsis                    | Hourly (within 6H) | Binary Classification |
| 4   | Kidney Function(KF)       | Once per stay | Regression |
| 5   | Length of Stay (LoS)      | Hourly (within 7D) | Regression |
## Model Types
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic+regression):
  Standard regression approach.
- [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html): Linear regression with combined L1 and L2 priors as regularizer.
- [LightGBM](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf): Efficient gradient
  boosting trees.
- [Long Short-term Memory (LSTM)](https://ieeexplore.ieee.org/document/818041): The most commonly used type of Recurrent Neural
  Networks for long sequences.
- [Gated Recurrent Unit (GRU)](https://arxiv.org/abs/1406.1078) : A extension to LSTM which showed improvements ([paper](https://arxiv.org/abs/1412.3555)).
- [Temporal Convolutional Networks (TCN)](https://arxiv.org/pdf/1803.01271 ): 1D convolution approach to sequence data. By
  using dilated convolution to extend the receptive field of the network it has shown great performance on long-term
  dependencies.
- [Transformers](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): The most common Attention
  based approach.
