import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from ...api import Explainer
from interpret import show
from interpret.blackbox import ShapKernel


class EBMShap(Explainer):
    """
    SHAP for InterpretML's EBM.

    model : model object
    data : np array
    mode : str, "tabular" or "images"
    """

    def __init__(self, model, data: torch.FloatTensor) -> None:

        self.output_dim = 2
        self.data = data.numpy()
        self.model = model

        super().__init__(model)

    def get_explanation(self, all_data: torch.FloatTensor, label: torch.FloatTensor, mode=None) -> torch.FloatTensor:

        # all_data = all_data.numpy()
        # label = label.numpy()
        # num_features = all_data.shape[1]

        shap = ShapKernel(predict_fn=self.model.predict_proba, data=self.data)
        shap_local = shap.explain_local(all_data, label)

        if mode == "graphic":
            return show(shap_local)
        
        else:
            #return shap_local
            res = []
            for i in range(len(label)):
                df  = pd.DataFrame(shap_local.data(i), columns = ['names', 'scores'])
                res.append(df['scores'].values.tolist())
                
            return torch.from_numpy(np.array(res))

        
        
'''
dir(object_name)


NOTE: When you want to pass data to test, pass just one datapoint, it's rlly slow
'''
