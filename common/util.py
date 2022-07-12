from common.env import *


class PATH:
    root   = abspath(dirname(dirname(__file__)))
    input  = join(root, 'data')
    output = join(root, 'output')
    target = join(root, 'info', 'target.csv')



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
