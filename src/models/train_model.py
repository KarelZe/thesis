import os
import random
import numpy as np

def set_seed(seed_val:int=42)->int:
    """
    Seeds basic parameters for reproducibility of results

    Args:
        seed_val (int, optional): random seed used in rngs. Defaults to 42.

    Returns:
        int: seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed_val)
    random.seed(seed_val)
    # pandas and numpy as discussed here: https://stackoverflow.com/a/52375474/5755604
    np.random.seed(seed_val)
    return seed_val
