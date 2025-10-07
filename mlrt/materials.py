# init 
import torch
import numpy as np
import math

class Material():
    """
    Optical materials for computing the refractive indices.
    
    The following follows the simple formula that

    n(\lambda) = A + B / \lambda^2

    where the two constants A and B can be computed from nD (index at 589.3 nm) and V (abbe number). 
    """
    def __init__(self, name=None):
        pass