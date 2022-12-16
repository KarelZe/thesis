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



def train_model(args, model: BaseModel, X_train: np.ndarray, X_val: np.ndarray,
                y_train: np.ndarray, y_val: np.ndarray) -> BaseModel:
    """ Train model using parameters args. 
        X_train, y_train: Training data and labels
        X_val and y_val: Test data and 
        :return: Trained model.
    """
    loss_history, val_loss_history = model.fit(X_train, y_train, X_val, y_val)
    val_model(model, X_val, y_val)
    return model


def global_removal_benchmark(args, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray,
                             y_val: np.ndarray, feature_importances: np.ndarray, order_morf=True) -> np.ndarray:
    """ Perform a feature removal benchmark for the attributions. 
        The features that are attributed the highest overall attribution scores are successivly removed from the 
        dataset. The model is then retrained.
        
        :param features_importances: A vector of D (number of features in X) values that contain the importance score for each feature.
            The features will be ordered by the absolute value of the passed importance.
        :param X_val: (N, D) train data (N samples, D features)
        :param y_val: (N) train class labels
        :param X_val: (M, D) test data (M samples, D features)
        :param y_val: (M) test class labels
        :param order_morf: Feature removal order. Either remove most important (morf=True) or least important (morf=False) features first
        :return: array with the obtained accuracies.
    """
    if X_train.shape[1] != len(feature_importances):
        raise ValueError("Number of Features in Trainset must be equal to number of importances passed.")

    ranking = np.argsort((1 if order_morf else -1) * np.abs(feature_importances))
    results = np.zeros(len(feature_importances))
    old_cat_index = args.cat_idx
    old_cat_dims = args.cat_dims
    for i in range(len(feature_importances)):
        remaining_features = len(feature_importances) - i
        use_idx = ranking[:remaining_features].copy()
        np.random.shuffle(use_idx)  # make sure the neighborhood relation is not important.

        print(f"Using {len(use_idx)} features ...")
        # Retrain the model and report acc.
        X_train_bench = X_train[:, use_idx]
        X_val_bench = X_val[:, use_idx]

        # modify feature args accordingly
        # args.num_features: points to the new number of features
        # args.cat_idx: Indices of categorical features
        # args.cat_dims: Number of categorical feature values
        # These values have to be recomputed for the modified dataset
        new_cat_idx = []
        new_cat_dims = []
        for j in range(len(use_idx)):
            if use_idx[j] in old_cat_index:
                old_index = old_cat_index.index(use_idx[j])
                new_cat_idx.append(j)
                new_cat_dims.append(old_cat_dims[old_index])

        args.cat_idx = new_cat_idx
        args.cat_dims = new_cat_dims
        args.num_features = remaining_features

        model_name = str2model(args.model_name)
        model = model_name(arguments.parameters[args.model_name], args)
        model = train_model(args, model, X_train_bench, X_val_bench, y_train, y_val)
        acc_obtained = val_model(model, X_val_bench, y_val)
        results[i] = acc_obtained

        res_dict = {}
        res_dict["model"] = args.model_name
        res_dict["order"] = "MoRF" if order_morf else "LeRF"
        res_dict["accuracies"] = results.tolist()
        res_dict["attributions"] = feature_importances.tolist()
    save_results_to_json_file(args, res_dict, f"global_benchmark{args.strategy}", append=True)
    # reset args to their old values.
    args.cat_idx = old_cat_index
    args.cat_dims = old_cat_dims
    return results


def compute_spearman_corr(attr1: np.ndarray, attr2: np.ndarray) -> np.ndarray:
    """ Compute the spearman rank correlations between two attributions. The attributions are first ranked 
        by their value. Pass absolute values, if you want to rank by magnitude only.
        Return a vector with the spearman correlation between all rows in the matrix.
        :param attr1: (N, D) attributions by method 1 (N samples, D features)
        :param attr2: (N, D) attributions by method 2 (N samples, D features)
        :return: (N) array with the rank correlation of the two attributions for each sample.
    """
    num_inputs = attr1.shape[0]
    resmat = np.zeros(num_inputs)
    ranks1 = np.argsort(np.argsort(attr1, axis=0), axis=0)
    ranks2 = np.argsort(np.argsort(attr2, axis=0), axis=0)

    cov = np.mean(ranks1 * ranks2, axis=0) - np.mean(ranks1, axis=0) * np.mean(ranks2, axis=0)  # E[XY]-E[Y]E[X]
    corr = cov / (np.std(ranks1, axis=0) * np.std(ranks2, axis=0))
    return corr


def compare_to_shap(args, attrs, model, X_val, sample_size=100):
    """ 
        Compare feature attributions by the model to shap values on a random set of validation points.
        Compute correlation and save raw output to JSON file.
        :param attrs: (N, D) model feature attributions
        :param model: The model to use.
        :param X_val: (N, D) test data (N samples, D features)
        :param sample_size: Number of points to choose
    """
    use_samples = np.arange(len(X_val))
    np.random.shuffle(use_samples)
    use_samples = use_samples[:sample_size]
    attrs = attrs[use_samples]

    res_dict = {}
    res_dict["model"] = args.model_name
    res_dict["model_attributions"] = attrs.tolist()

    shap_attrs = get_shap_attributions(model, X_val[use_samples])
    # save_attributions_image(attrs, feature_names, args.model_name+"_shap")
    res_dict["shap_attributions"] = shap_attrs.tolist()

    rank_corrs = compute_spearman_corr(np.abs(attrs), np.abs(shap_attrs))
    res_dict["rank_corr_mean"] = np.mean(rank_corrs)
    res_dict["rank_corr_std"] = np.std(rank_corrs)
    save_results_to_json_file(args, res_dict, f"shap_compare{args.strategy}", append=True)


def val_model(model: BaseModel, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """ 
        Validation of a trained classification model on the test set (X_val, y_val). 
        :param X_val: (N, D) test data (N samples, D features)
        :param y_val: (N) test class labels
        :return: accuracy
    """
    ypred = model.predict(X_val)
    if len(ypred.shape) == 2:
        ypred = ypred[:, -1]
    acc = np.sum((ypred.flatten() > 0.5) == y_val) / len(y_val)
    print("Accuracy: ", acc)
    return acc


def save_attributions_image(attrs: np.ndarray, namelist: tp.Optional[tp.List[str]] = None,
                            file_name: str = ""):
    """ Save attributions in a plot. 
        :param attrs: (N, D) attributions (N samples, D features)
        :param namelist: List of length D with column names
        :return: predicted labels of test data
    """
    attrs_abs = np.abs(attrs)
    attrs_abs -= np.min(attrs_abs)
    attrs_abs /= np.max(attrs_abs)
    plt.ioff()
    plt.matshow(attrs_abs)
    if namelist:
        plt.xticks(np.arange(len(namelist)), namelist, rotation=90)
    plt.tight_layout()
    plt.gcf().savefig(f"output/attributions_{file_name}.png")


def main(args):
    if args.model_name == "TabTransformer":  # Use discretized version of adult dataset for TabNet attributions.
        args.scale = False

    # Load dataset (currently only tested for the Adult data set)
    X, y = load_data(args)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=args.seed)

    # Create the model
    model_name = str2model(args.model_name)
    model = model_name(arguments.parameters[args.model_name], args)
    # Obtain a trained model to get attributions
    modelref = train_model(args, model, X_train, X_val, y_train, y_val)
    # Get attributions
    attrs = modelref.attribute(X_val, y_val, args.strategy)

    # Save the first 20 attributions to file.
    if args.dataset == "Adult" or args.dataset == "AdultCat":
        feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country']
    else:
        feature_names = None
    res_dict = {}
    res_dict["model"] = args.model_name
    res_dict["strategy"] = str(args.strategy)
    res_dict["dataset"] = args.dataset
    res_dict["attributions"] = attrs.tolist()
    save_results_to_json_file(args, res_dict, f"attributions{args.strategy}", append=True)
    save_attributions_image(attrs[:20, :], feature_names, args.model_name)

    # Run global attribution benchmark if flag is passed.
    if args.globalbenchmark:
        for order in [True, False]:
            for run in range(args.numruns):
                global_removal_benchmark(args, X_train, X_val, y_train, y_val, attrs.mean(axis=0).flatten(),
                                         order_morf=order)

    # Compute Shaples values and compare to model intrinsic attribution if flag is passed.
    if args.compareshap:
        compare_to_shap(args, attrs, modelref, X_val, sample_size=250)


if __name__ == "__main__":
    parser = get_attribution_parser()
    arguments = parser.parse_args()
    main(arguments)