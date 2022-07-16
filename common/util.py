from common.env import *


class PATH:
    root   = abspath(dirname(dirname(__file__)))
    input  = join(root, 'data')
    output = join(root, 'output')
    target = join(root, 'data', 'target.csv')


@dataclass
class Timer(ContextDecorator):
    """Context manager for timing the execution of a block of code.

    Parameters
    ----------
    name : str
        Name of the timer.

    Examples
    --------
    >>> from time import sleep
    >>> from analysis_tools.common.util import Timer
    >>> with Timer('Code1'):
    ...     sleep(1)
    ...
    * Code1: 1.00s (0.02m)
    """
    name: str = ''
    def __enter__(self):
        """Start timing the execution of a block of code.
        """
        self.start_time = time()
        return self
    def __exit__(self, *exc):
        """Stop timing the execution of a block of code.

        Parameters
        ----------
        exc : tuple
            Exception information.(dummy)
        """
        elapsed_time = time() - self.start_time
        print(f"* {self.name}: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        return False


def check_nan(data, name):
    """Print number of data and nan rows

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    name : str
        Name to identify data
    """
    print("* Data name:", name)
    print("  - Number of data:", len(data))
    print("  - Number of nan rows:", sum(data.isna().sum(axis='columns') > 0))

def set_random_seed(seed):
    """Set random seed for reusability

    seed : int
        Random seed
    """
    import tensorflow as tf
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def visualize_result(y_true, y_pred, n_rows=15, n_cols=15, ylim=None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(40, 20))
    # idxs = np.random.choice(len(y_true), len(axes.flatten()))
    idxs = np.arange(n_rows * n_cols)
    for idx, ax in zip(idxs, axes.flatten()):
        pd.DataFrame({'true': y_true[idx], 'pred': y_pred[idx]}).plot(ax=ax, ylim=ylim)
        ax.set_xticklabels([])
        # ax.set_yticklabels([])
    fig.tight_layout()
