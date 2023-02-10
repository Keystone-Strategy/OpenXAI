# Utils
import torch
import numpy as np
import pickle
from sklearn.metrics import auc

# ML models
# from xgboost import XGBClassifier
import xgboost as xgb

from openxai.LoadModel import LoadModel

# Data loaders
from openxai.dataloader import return_loaders

# Explanation models
from openxai.Explainer import Explainer

# Evaluation methods
from openxai.evaluator import Evaluator

# Perturbation methods required for the computation of the relative stability metrics
from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from openxai.explainers.catalog.perturbation_methods import NewDiscrete_NormalPerturbation

# Choose the model and the data set you wish to generate explanations for
data_loader_batch_size = 32
data_name = 'compas'  # must be one of ['heloc', 'adult', 'german', 'compas']
model_name = 'lr'  # must be one of ['lr', 'ann']

"""### (1) Data Loaders"""

# Get training and test loaders
loader_train, loader_test = return_loaders(data_name=data_name,
                                           download=True,
                                           batch_size=data_loader_batch_size)
data_iter = iter(loader_test)
inputs, labels = next(data_iter)
labels = labels.type(torch.int64)

print(type(inputs))

# get full training data set
data_all = torch.FloatTensor(loader_train.dataset.data)
labels_all = torch.FloatTensor(loader_train.dataset.targets)

# train an xgboost model
model = xgb.XGBClassifier()
model.fit(data_all, labels_all)
print(model.predict(inputs))
print(model.predict_proba(inputs))
#
print(model)
#
# # make predictions for test data
y_pred = model.predict(inputs)
predictions = [round(value) for value in y_pred]
print(predictions)

shap = Explainer(method='shap',
                 model=model,
                 dataset_tensor=data_all,
                 param_dict_shap=None)

shap_default_exp = shap.get_explanation(inputs.float(), label=labels)


explainers = [shap]
explanations = [shap_default_exp]
algos = ['shap']

def generate_mask(explanation, top_k):
    mask_indices = torch.topk(explanation, top_k).indices
    mask = torch.zeros(explanation.shape) > 10
    for i in mask_indices:
        mask[i] = True
    return mask


# Perturbation class parameters
perturbation_mean = 0.0
perturbation_std = 0.05
perturbation_flip_percentage = 0.03
if data_name == 'compas':
    feature_types = ['c', 'd', 'c', 'c', 'd', 'd', 'd']
# Adult feature types
elif data_name == 'adult':
    feature_types = ['c'] * 6 + ['d'] * 7
# Gaussian feature types
elif data_name == 'synthetic':
    feature_types = ['c'] * 20
# Heloc feature types
elif data_name == 'heloc':
    feature_types = ['c'] * 23
elif data_name == 'german':
    feature_types = ['c'] * 8 + ['d'] * 12

# Perturbation methods
if data_name == 'german':
    # use special perturbation class
    perturbation = NewDiscrete_NormalPerturbation("tabular",
                                                  mean=perturbation_mean,
                                                  std_dev=perturbation_std,
                                                  flip_percentage=perturbation_flip_percentage)

else:
    perturbation = NormalPerturbation("tabular",
                                      mean=perturbation_mean,
                                      std_dev=perturbation_std,
                                      flip_percentage=perturbation_flip_percentage)


"""### (4) Choose an evaluation metric"""

for explainer, explanation_x, algo in zip(explainers, explanations, algos):
    # PRA_AUC = []
    # RC_AUC = []
    # FA_AUC = []
    # RA_AUC = []
    # SA_AUC = []
    # SRA_AUC = []
    PGU_AUC = []
    PGI_AUC = []
    for index in range(data_loader_batch_size):
        print('iteration:', index)

        input_dict = dict()

        # inputs and models
        input_dict['x'] = inputs[index].reshape(-1)
        # print(input_dict['x'])
        input_dict['input_data'] = inputs
        input_dict['explainer'] = explainer
        # print(explainer)
        input_dict['explanation_x'] = explanation_x[index, :].flatten()
        # print(input_dict['explanation_x'])
        input_dict['model'] = model

        # perturbation method used for the stability metric
        input_dict['perturbation'] = perturbation
        input_dict['perturb_method'] = perturbation
        input_dict['perturb_max_distance'] = 0.4
        input_dict['feature_metadata'] = feature_types
        input_dict['p_norm'] = 2
        input_dict['eval_metric'] = None

        # gt label and model prediction
        input_dict['y'] = labels[index].detach().item()
        input_dict['y_pred'] = model.predict(inputs[index].unsqueeze(0).float())[0]

        # required for the representation stability measure
        input_dict['L_map'] = model

        # PRA = []
        # RC = []
        # FA = []
        # RA = []
        # SA = []
        # SRA = []
        PGU = []
        PGI = []
        # RIS = []
        # ROS = []
        # RRS = []

        auc_x = np.arange(1, input_dict['explanation_x'].shape[0]+1) / input_dict['explanation_x'].shape[0]

        print("y pred", input_dict['y_pred'])

        # topk and mask
        for topk in range(1, input_dict['explanation_x'].shape[0] + 1):
            input_dict['top_k'] = topk
            input_dict['mask'] = generate_mask(input_dict['explanation_x'].reshape(-1), input_dict['top_k'])

            evaluator = Evaluator(input_dict,
                                inputs=inputs,
                                labels=labels,
                                model=model,
                                explainer=shap,
                                model_type='xgb')

            # evaluate prediction gap on unimportant features
            PGU.append(evaluator.evaluate(metric='PGU'))
            print('PGU:', PGU)
            # evaluate prediction gap on important features
            PGI.append(evaluator.evaluate(metric='PGI'))
            print('PGI:', PGI[-1])
        # print(auc_x, PGU)
        PGU_AUC.append(auc(auc_x, PGU))
        PGI_AUC.append(auc(auc_x, PGI))

    print('--- MEAN ---')
    print('PGU', np.mean(PGU_AUC))
    print('PGI', np.mean(PGI_AUC))
    print('--- STD ---')
    print('PGU', np.std(PGU_AUC))
    print('PGI', np.std(PGI_AUC))