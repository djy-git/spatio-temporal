from common.env import *


class PATH:
    root   = abspath(dirname(dirname(__file__)))
    input  = join(root, 'data')
    output = join(root, 'output')
    target = join(root, 'info', 'target.csv')
