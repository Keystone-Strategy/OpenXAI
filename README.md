# OpenXAI : Towards a Transparent Evaluation of Model Explanations

----

[**Website**](https://open-xai.github.io/) | [**arXiv Paper**](https://arxiv.org/submit/4363609/preview)

**OpenXAI** is the first general-purpose lightweight library that provides a comprehensive list of functions to systematically evaluate the quality of explanations generated by attribute-based explanation methods. OpenXAI suppoerts the development of new datasets (both synthetic and real-world) and explanation methods, with a strong bent towards promoting systematic, reproducible, and transparenct evaluation of explanation methods.

OpenXAI is an open-source initiative that comprises of a collection of curated high-stakes datasets, models, and evaluation metrics, and provides a simple and easy-to-use API that enables researchers and practitioners to benchmark explanation methods using just a few lines of code.


## Updates
- `0.0.0`: OpenXAI is live! Now, you can submit the result for benchmarking an post-hoc explanation method on an evaluation metric. Checkout [here](https://open-xai.github.io/quick-start)!
- OpenXAI white paper is on [arXiv](https://arxiv.org/submit/4363609/preview)!


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

The core data loaders are lightweight with minimum dependency on external packages:

```bash
numpy, pandas, scikit-learn, captum
```


## Design of OpenXAI

OpenXAI is an open-source ecosystem comprising XAI-ready datasets, implementations of state-of-the-art explanation methods, evaluation metrics, leaderboards and documentation to promote transparency and collaboration around evaluations of post hoc explanations. OpenXAI can readily be used to *benchmark* new explanation methods as well as incorporate them into our framework and leaderboards. By enabling *systematic and efficient evaluation* and benchmarking of existing and new explanation methods, OpenXAI can inform and accelerate new research in the emerging field of XAI.

### OpenXAI DataLoaders

OpenXAI provides a Dataloader class that can be used to load the aforementioned collection of synthetic and real-world datasets as well as any other custom datasets, and ensures that they are XAI-ready. More specifically, this class takes as input the name of an existing OpenXAI dataset or a new dataset (name of the .csv file), and outputs a train set which can then be used to train a predictive model, a test set which can be used to generate local explanations of the trained model, as well as any ground-truth explanations (if and when available). If the dataset already comes with pre-determined train and test splits, this class loads train and test sets from those pre-determined splits. Otherwise, it divides the entire dataset randomly into train (70%) and test (30%) sets. Users can also customize the percentages of train-test splits.

For a concrete example, the code snippet below shows how to import the Dataloader class and load an existing OpenXAI dataset:

```python
from openxai import Dataloader
loader_train, loader_test = Dataloader.return_loaders(data_name=‘german’, download=True)
# get an input instance from the test dataset
inputs, labels = iter(loader_test).next()
```

### OpenXAI Pre-trained models

We also pre-trained two classes of predictive models (e.g., deep neural networks of varying degrees of complexity, logistic regression models etc.) and incorporated them into the OpenXAI framework so that they can be readily used for benchmarking explanation methods. The code snippet below shows how to load OpenXAI’s pre-trained models using our LoadModel class.

```python
from openxai import LoadModel
model = LoadModel(data_name=‘german’, ml_model=‘ann’)
```

Adding additional pre-trained models into the OpenXAI framework is as simple as uploading a file with details about model architecture and parameters in a specific template. Users can also submit requests to incorporate custom pre-trained models into the OpenXAI framework by filling a simple form and providing details about model architecture and parameters.

### OpenXAI Explainers

All the explanation methods included in OpenXAI are readily accessible through the *Explainer* class, and users just have to specify the method name in order to invoke the appropriate method and generate explanations as shown in the above code snippet. Users can easily incorporate their own custom explanation methods into the OpenXAI framework by extending the *Explainer* class and including the code for their methods in the *get_explanations* function of this class.

```python
from openxai import Explainer
exp_method = Explainer(method=‘LIME’)
explanations = exp_method.get_explanations(model, X=inputs, y=labels)
```

Users can then submit a request to incorporate their custom methods into OpenXAI library by filling a form and providing the GitHub link to their code as well as a summary of their explanation method.

### OpenXAI Evaluation

Benchmarking an explanation method using evaluation metrics is quite simple and the code snippet below describes how to invoke the RIS metric. Users can easily incorporate their own custom evaluation metrics into OpenXAI by filling a form and providing the GitHub link to their code as well as a summary of their metric. Note that the code should be in the form of a function which takes as input data instances, corresponding model predictions and their explanations, as well as OpenXAI’s model object and returns a numerical score. 

```python
from openxai import Evaluator
metric_evaluator = Evaluator(inputs, labels, model, explanations)
score = metric_evaluator.eval(metric=‘RIS’)
```

### OpenXAI Leaderboards

Every explanation method in OpenXAI is a benchmark, and we provide dataloaders, pre-trained models, together with explanation methods and performance evaluation metrics. To participate in the leaderboard for a specific benchmark, follow these steps:

* Use the OpenXAI benchmark dataloader to retrieve a given dataset.

* Use the OpenXAI LoadModel to load a pre-trained model.

* Use the OpenXAI Explainer to load a post hoc explanation method.

* Submit the performance of the explanation method for a given metric.

As many datasets share a therapeutics theme, we organize benchmarks into meaningfully defined groups, which we refer to as benchmark groups. Datasets and tasks within a benchmark group are carefully curated and centered around a theme (for example, TDC contains a benchmark group to support ML predictions of the ADMET properties). While every benchmark group consists of multiple benchmarks, it is possible to separately submit results for each benchmark in the group. Here is the code framework to access the benchmarks:

## Cite Us

If you find OpenXAI benchmark useful, cite our paper:

```
@article{agarwal2022openxai,
  title={OpenXAI: Towards a Transparent Evaluation of Model Explanations},
  author={Agarwal, Chirag and Saxena, Eshika and Krishna, Satyapriya and Pawelczyk, Martin and Johnson, Nari and Puri, Isha and Zitnik, Marinka and Lakkaraju, Himabindu},
  journal={arXiv},
  year={2022}
}
```

## Contact

Reach us at [openxaibench@gmail.com](mailto:openxaibench@gmail.com) or open a GitHub issue.

## License
OpenXAI codebase is under MIT license. For individual dataset usage, please refer to the dataset license found in the website.
