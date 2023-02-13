import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from ... api import Explainer
from interpret import show


class EBM(Explainer):
    """
    EBM feature importance

    model : model object
    data : np array
    mode : str, "tabular" or "images"
    """

    def __init__(self, model, data=None) -> None:

        self.output_dim = 2
        self.model = model

        super(EBM, self).__init__(model)

    def get_explanation(self, all_data = None, label=None, mode=None) -> torch.FloatTensor:
        # all_data = all_data.numpy()
        # num_features = all_data.shape[1]

        ebm_res = self.model.explain_global()

        if mode == "graphic":
            return show(ebm_res)
        else:
            df = pd.DataFrame (ebm_res.data(), columns = ['names','scores'])
            fin_res = df[~df["names"].str.contains("&")]
            return torch.FloatTensor(fin_res['scores'])
