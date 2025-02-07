# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch

from .odesolvers import ODEsolver, ODEsolverRegistry

import numpy as np
import matplotlib.pyplot as plt
import random

__all__ = [
    'ODEsolverRegistry', 'ODEsolver', 'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_white_box_solver(
    odesolver_name,  ode, VF_fn, Y, Y_prior=None,
    T_rev=1.0, t_eps=0.03, N=4,  **kwargs
):
   
    odesolver_cls = ODEsolverRegistry.get_by_name(odesolver_name)
    
    odesolver = odesolver_cls(ode, VF_fn)

    def ode_solver(Y_prior=Y_prior):
        """The PC sampler function."""
        with torch.no_grad():
            
            if Y_prior == None:
                Y_prior = Y
            
            xt, _ = ode.prior_sampling(Y_prior.shape, Y_prior)
            xt = xt.to(Y_prior.device)
            timesteps = torch.linspace(T_rev, t_eps, N, device=Y.device)
            ns = len(timesteps)
            
            if VF_fn.__class__.__name__ == "VFModel_Finetuning_SGMSE_CRP":
                print("sampling 잘되니?")
                timesteps = torch.linspace(T_rev, t_eps, VF_fn.N_epsilon, device=Y.device)
                for i in range(len(timesteps)):
                    t = timesteps[i]
                    if i != len(timesteps) - 1:
                        stepsize = t - timesteps[i+1]
                    else:
                        stepsize = timesteps[-1]
                        
                    vec_t = torch.ones(Y.shape[0], device=Y.device) * t
                                
                    xt = odesolver.update_fn(xt, vec_t, Y, stepsize)
                    
                x_result = xt
                return x_result, ns
            elif VF_fn.__class__.__name__ == "VFModel_Finetuning":
                #random 샘플링을 이용한 샘플링, y와 예측치 s_hat 간에
                if N!=0:
                    timesteps = torch.linspace(T_rev, t_eps, N, device=Y.device)
                    random_number = random.uniform(0, 1)
                    # print("N!=0")
                    shat =xt- VF_fn(xt,torch.ones(Y.shape[0],device=Y.device), Y) 
                    x1 = VF_fn.weight_shat * shat + (1-VF_fn.weight_shat) * Y 
                
                    for i in range(len(timesteps)):
                        t = timesteps[i]
                        if i != len(timesteps) - 1:
                            stepsize = t - timesteps[i+1]
                        else:
                            stepsize = timesteps[-1]
                            
                        vec_t = torch.ones(Y.shape[0], device=Y.device) * t
                                    
                        xt = odesolver.update_fn(xt, vec_t, x1, stepsize)
                elif N==0:
                    xt =xt- VF_fn(xt,torch.ones(Y.shape[0],device=Y.device), Y) 
                x_result = xt
                if N!=0:
                    ns = len(timesteps)
                else:
                    ns = 0
            
                return x_result, ns
            else:
                raise("model의 이름을 다시 확인해봐")
    
    return ode_solver
