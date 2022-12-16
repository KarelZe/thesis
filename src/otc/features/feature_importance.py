# adapted from:
# https://github.com/kathrinse/TabSurvey/blob/main/utils/baseline_attributions.py

from tkinter.tix import X_REGION
import captum
import numpy as np
from models.basemodel import BaseModel
import shap


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def get_probabilistic_predictions(model: BaseModel, X: np.ndarray):
    """ Return output probabilities as a single vector. 
        Note: will be removed when predict interface is remade.
    """
    ypred = model.predict(X)
    if len(ypred.shape) == 2:
        ypred = ypred[:, -1]
    return ypred


def get_shap_attributions(model: BaseModel, X: np.ndarray):
    """ Return local KernelSHAP attributions for the data. 
        :param model: The model to generate attributions for:
        :param X: the data that the attributions are generated for (N, D)-array.
        retunr (N,D) KernelShap attributions for each feature.
    """
    f = lambda x: get_probabilistic_predictions(model, x)
    kernelshap = shap.KernelExplainer(f, shap.sample(X, 50))
    shap_values = kernelshap.shap_values(X, nsamples=1000)  # nsamples = no. of feature coalitions
    print(shap_values.shape, shap_values.dtype)
    return shap_values



# copied from https://github.com/kathrinse/TabSurvey
# feature attributions for TabNet
    def attribute(self, X: np.ndarray, y: np.ndarray, stategy=""):
        """ Generate feature attributions for the model input.
            Only strategy are supported: default ("") 
            Return attribution in the same shape as X.
        """
        X = np.array(X, dtype=np.float)
        attributions = self.model.explain(torch.tensor(X, dtype=torch.float32))[0]
        return attributions

# feaure attributions for TabTransformer
# copied from https://github.com/kathrinse/TabSurvey
    def attribute(self, X: np.ndarray, y: np.ndarray, strategy=""):
        """ Generate feature attributions for the model input.
            Two strategies are supported: default ("") or "diag". The default strategie takes the sum
            over a column of the attention map, while "diag" returns only the diagonal (feature attention to itself)
            of the attention map.
            return array with the same shape as X. The number of columns is equal to the number of categorical values in X.
        """
        X = np.array(X, dtype=np.float)
        # Unroll and Rerun until first attention stage.

        X = torch.tensor(X).float()

        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                 num_workers=2)

        attentions_list = []
        with torch.no_grad():
            for batch_X in test_loader:
                x_categ = batch_X[0][:, self.args.cat_idx].int().to(self.device) if self.args.cat_idx else None
                x_cont = batch_X[0][:, self.num_idx].to(self.device)
                if x_categ is not None:
                    x_categ += self.model.categories_offset
                    # Tranformer
                    x = self.model.transformer.embeds(x_categ)
                    
                    # Prenorm.
                    x = self.model.transformer.layers[0][0].fn.norm(x)

                    # Attention
                    active_transformer =  self.model.transformer.layers[0][0].fn.fn
                    h = active_transformer.heads
                    q, k, v = active_transformer.to_qkv(x).chunk(3, dim=-1)
                    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
                    sim = einsum('b h i d, b h j d -> b h i j', q, k) * active_transformer.scale
                    attn = sim.softmax(dim=-1) 
                    if strategy == "diag":
                        print(attn.shape)
                        attentions_list.append(attn.diagonal(0,2,3))
                    else:
                        attentions_list.append(attn.sum(dim=1))
                else:
                    raise ValueError("Attention can only be computed for categorical values in TabTransformer.")
            attentions_list = torch.cat(attentions_list).sum(dim=1)
        return attentions_list.numpy()