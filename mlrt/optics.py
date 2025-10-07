import math
import numpy as np
import torch
import pathlib
from .basics import *
class Lensgroup():
    """
    The origin of the Lensgroup, which is a collection of multiple optical surfaces, is located at "origin".
    The Lensgroup can rotate freely around the x/y axes, and the rotation angles are defined as "theta" (x,y,z) (in degrees).
    
    In the Lensgroup's coordinate system, which is the object frame coordinate system, surfaces are arranged starting from "z = 0".
    There is a small 3D origin shift, called "shift", between the center of the surface (0,0,0) and the mount's origin.
    The sum of the shift and the origin is equal to the Lensgroup's origin.
    
    pixel size: sensor properties [um]
    file size: resolution of the sensor [pixel]
    """
    def __init__(self, origin = torch.zeros(3), shift=torch.zeros(3), pixel_size=6.45,  film_size=[512,512],theta=torch.zeros(3),device=torch.device('cpu')):
        self.origin = origin.to(device)
        self.shift = shift.to(device)
        self.theta = theta.to(device)
        self.device = device
        
        # sequential of lenes
        self.surfaces = []
        self.materials = []

        # sensor 
        self.pixel_size = pixel_size
        self.film_size = film_size
        
    def _compute_transformation(self, _x=0.0, _y=0.0, _z=0.0):
        # we compute to_world transformation given the input positional parameters (angles)
        R = ( rodrigues_rotation_matrix(torch.Tensor([1, 0, 0]).to(self.device), torch.deg2rad(self.theta_x+_x)) @ 
              rodrigues_rotation_matrix(torch.Tensor([0, 1, 0]).to(self.device), torch.deg2rad(self.theta_y+_y)) @ 
              rodrigues_rotation_matrix(torch.Tensor([0, 0, 1]).to(self.device), torch.deg2rad(self.theta_z+_z)) )
        t = self.origin + R @ self.shift
        return Transformation(R, t)
    
