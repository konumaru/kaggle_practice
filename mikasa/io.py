import os
import pickle


def load_pickle(filepath, verbose: bool = True):
    if verbose:
        print(f"Load pickle from {filepath}.")
    with open(filepath, "rb") as file:
        return pickle.load(file)


def dump_pickle(data, filepath, verbose: bool = True):
    if verbose:
        print(f"Dump pickle to {filepath}.")
    with open(filepath, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def cache_result(filepath, use_cache=True):
    """Save result decorator.

    Parameters
    ----------
    filename : str
        filename, when save with pickle.
    use_cache : bool, optional
        Is use already cash result then pass method process, by default True
    """

    def _acept_func(func):
        def run_func(*args, **kwargs):
            if use_cache and os.path.exists(filepath):
                print(f"Load Cached data, {filepath}")
                return load_pickle(filepath)
            result = func(*args, **kwargs)

            print(f"Cache to {filepath}")
            dump_pickle(result, filepath)
            return result

        return run_func

    return _acept_func
