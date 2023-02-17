# OpenXAI : Towards a Transparent Evaluation of Model Explanations

----

## KS: Notes
- Some features of this distribution of OpenXAI may not work with a torch model. The PGU/PGI computation has been modified to take into account the EBM's predict_proba() function. 
- ebm_global(), ebm_local() and InterpretML's SHAP have been added to this distribution of OpenXAI by extending the in-built Explainer class.
- You can find a test notebook that demonstrates this integration (*OpenXAI+EBM.ipynb*) under the KS_Notebooks folder.

 
[**Website**](https://open-xai.github.io/) | [**arXiv Paper**](https://arxiv.org/abs/2206.11104)

**OpenXAI** is the first general-purpose lightweight library that provides a comprehensive list of functions to systematically evaluate the quality of explanations generated by attribute-based explanation methods. OpenXAI supports the development of new datasets (both synthetic and real-world) and explanation methods, with a strong bent towards promoting systematic, reproducible, and transparent evaluation of explanation methods.

OpenXAI is an open-source initiative that comprises of a collection of curated high-stakes datasets, models, and evaluation metrics, and provides a simple and easy-to-use API that enables researchers and practitioners to benchmark explanation methods using just a few lines of code.


## Updates
- `0.0.0`: OpenXAI is live! Now, you can submit the result for benchmarking an post-hoc explanation method on an evaluation metric. Checkout [here](https://open-xai.github.io/quick-start)!
- OpenXAI white paper is on [arXiv](https://arxiv.org/abs/2206.11104)!


## Unique Features of OpenXAI
- *Diverse areas of XAI research*: OpenXAI includes ready-to-use API interfaces for seven state-of-the-art feature attribution methods and 22 metrics to quantify their performance. Further, it provides a flexible synthetic data generator to synthesize datasets of varying sizes, complexity, and dimensionality that facilitate the construction of ground truth explanations and a comprehensive collection of real-world datasets.
- *Data functions*: OpenXAI provides extensive data functions, including data evaluators, meaningful data splits, explanation methods, and evaluation metrics.
- *Leaderboards*: OpenXAI provides the first ever public XAI leaderboards to promote transparency, and to allow users to easily compare the performance of multiple explanation methods.
- *Open-source initiative*: OpenXAI is an open-source initiative and easily extensible.

## Installation

### Using `pip`

To install the core environment dependencies of OpenXAI, use `pip` by cloning the OpenXAI repo into your local environment:

```bash
pip install -e . 
```

## Design of OpenXAI

OpenXAI is an open-source ecosystem comprising XAI-ready datasets, implementations of state-of-the-art explanation methods, evaluation metrics, leaderboards and documentation to promote transparency and collaboration around evaluations of post hoc explanations. OpenXAI can readily be used to *benchmark* new explanation methods as well as incorporate them into our framework and leaderboards. By enabling *systematic and efficient evaluation* and benchmarking of existing and new explanation methods, OpenXAI can inform and accelerate new research in the emerging field of XAI.

### OpenXAI DataLoaders

OpenXAI provides a Dataloader class that can be used to load the aforementioned collection of synthetic and real-world datasets as well as any other custom datasets, and ensures that they are XAI-ready. More specifically, this class takes as input the name of an existing OpenXAI dataset or a new dataset (name of the .csv file), and outputs a train set which can then be used to train a predictive model, a test set which can be used to generate local explanations of the trained model, as well as any ground-truth explanations (if and when available). If the dataset already comes with pre-determined train and test splits, this class loads train and test sets from those pre-determined splits. Otherwise, it divides the entire dataset randomly into train (70%) and test (30%) sets. Users can also customize the percentages of train-test splits.

For a concrete example, the code snippet below shows how to import the Dataloader class and load an existing OpenXAI dataset:

```python
from openxai.dataloader import return_loaders
loader_train, loader_test = return_loaders(data_name=‘german’, download=True)
# get an input instance from the test dataset
inputs, labels = iter(loader_test).next()
```

### OpenXAI Pre-trained models

We also pre-trained two classes of predictive models (e.g., deep neural networks of varying degrees of complexity, logistic regression models etc.) and incorporated them into the OpenXAI framework so that they can be readily used for benchmarking explanation methods. The code snippet below shows how to load OpenXAI’s pre-trained models using our LoadModel class.

```python
from openxai import LoadModel
model = LoadModel(data_name= 'german', ml_model='ann', pretrained=True)
```

Adding additional pre-trained models into the OpenXAI framework is as simple as uploading a file with details about model architecture and parameters in a specific template. Users can also submit requests to incorporate custom pre-trained models into the OpenXAI framework by filling a simple form and providing details about model architecture and parameters.

### OpenXAI Explainers

All the explanation methods included in OpenXAI are readily accessible through the *Explainer* class, and users just have to specify the method name in order to invoke the appropriate method and generate explanations as shown in the above code snippet. Users can easily incorporate their own custom explanation methods into the OpenXAI framework by extending the *Explainer* class and including the code for their methods in the *get_explanations* function of this class.

```python
from openxai import Explainer
exp_method = Explainer(method= 'lime',model=model, dataset_tensor=inputs)
explanations= exp_method.get_explanation(inputs, labels)
```

Users can then submit a request to incorporate their custom methods into OpenXAI library by filling a form and providing the GitHub link to their code as well as a summary of their explanation method.

### OpenXAI Evaluation

Benchmarking an explanation method using evaluation metrics is quite simple and the code snippet below describes how to invoke the RIS metric. Users can easily incorporate their own custom evaluation metrics into OpenXAI by filling a form and providing the GitHub link to their code as well as a summary of their metric. Note that the code should be in the form of a function which takes as input data instances, corresponding model predictions and their explanations, as well as OpenXAI’s model object and returns a numerical score. Finally, the input_dict is described [here](https://github.com/AI4LIFE-GROUP/OpenXAI/blob/main/OpenXAI%20quickstart.ipynb).

```python
from openxai import Evaluator
metric_evaluator = Evaluator(input_dict, inputs, labels, model, exp_method)
score = metric_evaluator.evaluate(metric='RIS')
```

### OpenXAI Metrics

#### Ground-truth Faithfulness
OpenXAI includes the following metrics to calculate the agreement between ground-truth explanations (i.e., coefficients of logistic regression models) and explanations generated by state-of-the-art methods.

1. `Feature Agreement (FA)` metric computes the fraction of top-K features that are common between a given post hoc explanation and the corresponding ground truth explanation.
2. `Rank Agreement (RA)` metric measures the fraction of top-K features that are not only common between a given post hoc explanation and the corresponding ground truth explanation, but also have the same position in the respective rank orders.
3. `Sign Agreement (SA)` metric computes the fraction of top-K features that are not only common between a given post hoc explanation and the corresponding ground truth explanation, but also share the same sign (direction of contribution) in both the explanations.
4. `Signed Rank Agreement (SRA)` metric computes the fraction of top-K features that are not only common between a given post hoc explanation and the corresponding ground truth explanation, but also share the same feature attribution sign (direction of contribution) and position (rank) in both the explanations.
5. `Rank Correlation (RC)` metric computes the Spearman’s rank correlation coefficient to measure the agreement between feature rankings provided by a given post hoc explanation and the corresponding ground truth explanation.
6. `Pairwise Rank Agreement (PRA)` metric captures if the relative ordering of every pair of features is the same for a given post hoc explanation as well as the corresponding ground truth explanation i.e., if feature A is more important than B according to one explanation, then the same should be true for the other explanation. More specifically, this metric computes the fraction of feature pairs for which the relative ordering is the same between the two explanations.

#### Predicted Faithfulness
OpenXAI includes two complementary predictive faithfulness metrics: i) `Prediction Gap on Important feature perturbation (PGI)` which measures the difference in prediction probability that results from perturbing the features deemed as influential by a given post hoc explanation, and ii) `Prediction Gap on Unimportant feature perturbation (PGU)` which measures the difference in prediction probability that results from perturbing the features deemed as unimportant by a given post hoc explanation.

#### Stability
OpenXAI incorporates three stability metrics: i) `Relative Input Stability (RIS)` which measure the maximum change in explanation relative to changes in the inputs, ii) `Relative Representation Stability (RRS)` which measure the maximum change in explanation relative to changes in the internal representation learned by the model, and iii) `Relative Output Stability (ROS)` which measure the maximum change in explanation relative to changes in output prediction probabilities.

#### Fairness
We report the average of all faithfulness and stability metric values across instances in the majority and minority subgroups, and then take the absolute difference between them to check if there are significant disparities.

### OpenXAI Leaderboards

Every explanation method in OpenXAI is a benchmark, and we provide dataloaders, pre-trained models, together with explanation methods and performance evaluation metrics. To participate in the leaderboard for a specific benchmark, follow these steps:

* Use the OpenXAI benchmark dataloader to retrieve a given dataset.

* Use the OpenXAI LoadModel to load a pre-trained model.

* Use the OpenXAI Explainer to load a post hoc explanation method.

* Submit the performance of the explanation method for a given metric.

## Cite Us

If you find OpenXAI benchmark useful, cite our paper:

```
@inproceedings{
agarwal2022openxai,
title={Open{XAI}: Towards a Transparent Evaluation of Model Explanations},
author={Chirag Agarwal and Satyapriya Krishna and Eshika Saxena and Martin Pawelczyk and Nari Johnson and Isha Puri and Marinka Zitnik and Himabindu Lakkaraju},
booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2022},
url={https://openreview.net/forum?id=MU2495w47rz}
}
```

## Contact

Reach us at [openxaibench@gmail.com](mailto:openxaibench@gmail.com) or open a GitHub issue.

## License
OpenXAI codebase is under MIT license. For individual dataset usage, please refer to the dataset license found on the website.
