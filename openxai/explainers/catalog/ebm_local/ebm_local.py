import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from ... api import Explainer
from interpret import show


class EBMLocal(Explainer):
    """
    EBM feature importance

    model : model object
    data : np array
    mode : str, "tabular" or "images"
    """

    def __init__(self, model, data: torch.FloatTensor) -> None:

        self.output_dim = 2
        self.data = data.numpy()
        self.model = model

        super(EBMLocal, self).__init__(model)

    def get_explanation(self, all_data: torch.FloatTensor, label: torch.FloatTensor, mode=None) -> torch.FloatTensor:

        # all_data = all_data.numpy()
        # label = label.numpy()
        # num_features = all_data.shape[1]

        ebmLocal_res = self.model.explain_local(all_data, label)

        if mode == "graphic":
            return show(ebmLocal_res)
        
        else:
            res = []
            for i in range(len(label)):
                df  = pd.DataFrame(ebmLocal_res.data(i), columns = ['names', 'scores'])
                fin_res = df[~df["names"].str.contains("&")]
                temp = []
                for j in range(len(fin_res['scores'])):
                    temp.append(fin_res['scores'][j])
                res.append(temp)
                temp = []

            # df = pd.DataFrame (ebm_res, columns = ['names','scores'])
            
            return torch.from_numpy(np.array(res))
