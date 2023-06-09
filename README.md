![YAIB logo](https://github.com/rvandewater/YAIB/blob/development/docs/figures/yaib_logo.png)
[![arXiv](https://img.shields.io/badge/arXiv-2306.05109-b31b1b.svg)](http://arxiv.org/abs/2306.05109)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
# 🧪 Yet Another ICU Benchmark (YAIB) - models
Models trained for the publication of [Yet Another ICU Benchmark](https://github.com/rvandewater/YAIB). The models are named in the following manner: _dataset_task_model_cv-repetition_cv-fold_. Please not that it is possible that the performance of the classification models might deviate slightly from the official paper results due to major improvements to YAIB in the meantime. We hope to confirm the results once YAIB is out of alpha.

The following repositories may be relevant as well:
- [YAIB](https://github.com/rvandewater/YAIB): The main YAIB repository.
- [YAIB-models](https://github.com/rvandewater/YAIB-models): Pretrained models for YAIB.
- [ReciPys](https://github.com/rvandewater/ReciPys): Preprocessing package for YAIB pipelines.

## Datasets
We support the following datasets out of the box:

| **Dataset**                 | [MIMIC-III](https://physionet.org/content/mimiciii/) / [IV](https://physionet.org/content/mimiciv/) | [eICU-CRD](https://physionet.org/content/eicu-crd/) | [HiRID](https://physionet.org/content/hirid/1.1.1/) | [AUMCdb](https://doi.org/10.17026/dans-22u-f8vd) |
|-------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|--------------------------------------------------|
| **Admissions**              | 40k / 73k                                                                                           | 200k                                                | 33k                                                 | 23k                                              |
| **Version**                 | v1.4 / v2.2                                                                                         | v2.0                                                | v1.1.1                                              | v1.0.2                                           |                                                     |                                                     |                                                  |
| **Frequency** (time-series) | 1 hour                                                                                              | 5 minutes                                           | 2 / 5 minutes                                       | up to 1 minute                                   |
| **Originally published**    | 2015  / 2020                                                                                        | 2017                                                | 2020                                                | 2019                                             |                                                                                                     |                                                     |                                                     |                                                  |
| **Origin**                  | USA                                                                                                 | USA                                                 | Switzerland                                         | Netherlands                                      |


## Tasks
| No  | Task Theme                | Frequency        | Type                                | 
|-----|---------------------------|--------------------|-------------------------------------|
| 1   | ICU Mortality             | Once per Stay (after 24H) | Binary Classification  |
| 2   | Acute Kidney Injury (AKI) | Hourly (within 6H) | Binary Classification |
| 3   | Sepsis                    | Hourly (within 6H) | Binary Classification |
| 4   | Kidney Function (KF)       | Once per stay | Regression |
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
 
# 📄Paper

To reproduce the benchmarks in our paper, we refer to: the [ML reproducibility document](https://github.com/rvandewater/YAIB/blob/development/PAPER.md).
If you use this code in your research, please cite the following publication:

```
@article{vandewaterYetAnotherICUBenchmark2023,
	title = {Yet Another ICU Benchmark: A Flexible Multi-Center Framework for Clinical ML},
	shorttitle = {Yet Another ICU Benchmark},
	url = {http://arxiv.org/abs/2306.05109},
	language = {en},
	urldate = {2023-06-09},
	publisher = {arXiv},
	author = {van de Water, Robin and Schmidt, Hendrik and Elbers, Paul and Thoral, Patrick and Arnrich, Bert and Rockenschaub, Patrick},
	month = jun,
	year = {2023},
	note = {arXiv:2306.05109 [cs]},
	keywords = {Computer Science - Machine Learning},
}
```
This paper can also be found on arxiv: https://arxiv.org/pdf/2306.05109.pdf

# Acknowledgements

We do not own any of the datasets used in this benchmark. This project uses heavily adapted components of
the [HiRID benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark/). We thank the authors for providing this codebase and
encourage further development to benefit the scientific community. The demo datasets have been released under
an [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).
