import numpy as np

# --- Model Parameters Class ---
class ModelParams:
    def __init__(self,
                 base_k3=0.4,
                 base_k4=0.4,
                 base_k5=0.4,
                 vegetation_weight=1.2,
                 density_weight=0.8,
                 shadow_gamma=1.5):
        self.base_k3 = base_k3
        self.base_k4 = base_k4
        self.base_k5 = base_k5
        self.vegetation_weight = vegetation_weight
        self.density_weight = density_weight
        self.shadow_gamma = shadow_gamma
        # self.buffer_radius = buffer_radius
        # self.buffer_extinction = buffer_extinction

