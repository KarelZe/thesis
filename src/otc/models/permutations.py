import torch


def uniform_replace(unique_elements, mapping):
    mod = unique_elements.shape[0]
    shift = torch.randint_like(mapping, 1, mod)
    new_mapping = (mapping + shift) % mod
    return new_mapping


class BasePermutations:
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def preprocess(self):
        pass

    def permute(self):
        pass

    def gen_permutations(self, part):
        pass


class ShufflePermutations(BasePermutations):
    def permute(self, X):
        if X is None:
            return None
        # generate random index array
        return torch.randint_like(X, X.shape[0], dtype=torch.long)

    def gen_permutations(self, part):
        # permute numerical and categorical by random index
        X_num = self.X_num[part]
        X_cat = self.X_cat[part] if self.X_cat else None
        return self.permute(X_num), self.permute(X_cat)


def gen_permutations_class(name, X_num, X_cat):
    if name == "shuffle":
        perm_class = ShufflePermutations(X_num, X_cat)
    else:
        raise ValueError("Unknown permutation type")

    perm_class.preprocess()
    return perm_class.gen_permutations
