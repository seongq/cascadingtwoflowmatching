import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from flowmse import sampling
from flowmse.odes import ODERegistry
from flowmse.backbones import BackboneRegistry
from flowmse.util.inference import evaluate_model
from flowmse.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt
# from flowmse.odes import OTFLOW
import random



class VFModel_Finetuning(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (0 by default)")
        parser.add_argument("--T_rev",type=float, default=1.0, help="The maximum time")
        
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        parser.add_argument("--mode", type=str, required=True, choices=["first-only", "second-only", "first-second","mse-second", "first-mse"])
        return parser

    def __init__(
        self, backbone, ode, lr=1e-4, ema_decay=0.999, t_eps=0.03, T_rev = 1.0,  loss_abs_exponent=0.5, 
        num_eval_files=10, loss_type='mse', data_module_cls=None, N_enh=10, enhancement=False, weight_shat=0.8,weight_y=0.2, mode=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        self.weight_shat = weight_shat
        self.weight_y = weight_y
        ode_cls = ODERegistry.get_by_name(ode)
        self.enhancement = enhancement
        self.N_enh = N_enh
        self.ode = ode_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.ode.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.mode = mode


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _mse_loss(self, x, x_hat):    
        err = x-x_hat
        losses = torch.square(err.abs())

        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    
    def _loss(self, vectorfield, condVF):    
        if self.loss_type == 'mse':
            err = vectorfield-condVF
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            err = vectorfield-condVF
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x0, y = batch
        rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps        
        t = torch.min(rdm, torch.tensor(self.T_rev))
        mean, std = self.ode.marginal_prob(x0, t, y)
        z = torch.randn_like(x0)  #
        sigmas = std[:, None, None, None]
        xt = mean + sigmas * z
        der_std = self.ode.der_std(t)
        der_mean = self.ode.der_mean(x0,t,y)
        condVF = der_std * z + der_mean    
        VECTORFIELD_origin = self(xt,t,y)
        loss_original_flow = self._loss(VECTORFIELD_origin,condVF)

        if self.mode =="second-only":

            x1, _ = self.ode.prior_sampling(y.shape,y)
            ENHANCER = self(x1,torch.ones(y.shape[0], device=y.device),y)
            ENHANCEMENT = x1 - ENHANCER
            
            
            CONDITION = 0.5 * ENHANCEMENT + 0.5 * y
            rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps        
            t = torch.min(rdm, torch.tensor(self.T_rev))
            mean, std = self.ode.marginal_prob(x0, t, ENHANCEMENT)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,ENHANCEMENT)
            condVF = der_std * z + der_mean    
            VECTORFIELD = self(xt,t,CONDITION)
            loss_flow = self._loss(VECTORFIELD,condVF)
            
            loss = loss_flow
            
        elif self.mode =="first-only":
            loss = loss_original_flow
            
        elif self.mode =="first-second":

            x1, _ = self.ode.prior_sampling(y.shape,y)
            ENHANCER = self(x1,torch.ones(y.shape[0], device=y.device),y)
            ENHANCEMENT = x1 - ENHANCER
            
            
            CONDITION = 0.5 * ENHANCEMENT + 0.5 * y
            rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps        
            t = torch.min(rdm, torch.tensor(self.T_rev))
            mean, std = self.ode.marginal_prob(x0, t, ENHANCEMENT)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,ENHANCEMENT)
            condVF = der_std * z + der_mean    
            VECTORFIELD = self(xt,t,CONDITION)
            loss_flow = self._loss(VECTORFIELD,condVF)
            
            loss = loss_original_flow + loss_flow
        elif self.mode=="mse-second":
            x1, _ = self.ode.prior_sampling(y.shape,y)
            ENHANCER = self(x1,torch.ones(y.shape[0], device=y.device),y)
            ENHANCEMENT = x1 - ENHANCER
            
            loss_enh = self._loss(ENHANCEMENT,x0)
            
            CONDITION = 0.5 * ENHANCEMENT + 0.5 * y
            rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps        
            t = torch.min(rdm, torch.tensor(self.T_rev))
            mean, std = self.ode.marginal_prob(x0, t, ENHANCEMENT)
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(x0,t,ENHANCEMENT)
            condVF = der_std * z + der_mean    
            VECTORFIELD = self(xt,t,CONDITION)
            loss_flow = self._loss(VECTORFIELD,condVF)
            
            loss = loss_enh + loss_flow
        elif self.mode=="first-mse":
            x1, _ = self.ode.prior_sampling(y.shape,y)
            ENHANCER = self(x1,torch.ones(y.shape[0], device=y.device),y)
            ENHANCEMENT = x1 - ENHANCER
            
            loss_enh = self._loss(ENHANCEMENT,x0)
            
            
            loss =  loss_original_flow +loss_enh 
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
    
    def add_para(self, weight_shat):
        self.weight_shat = weight_shat
        self.weight_y = 1-self.weight_shat
       




class VFModel_Finetuning_SGMSE_CRP(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (0 by default)")
        parser.add_argument("--T_rev",type=float, default=1.0, help="The maximum time")
        
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        
        return parser

    def __init__(
        self, backbone, ode, lr=1e-4, ema_decay=0.999, t_eps=0.03, T_rev = 1.0,  loss_abs_exponent=0.5, 
        num_eval_files=10, loss_type='mse', data_module_cls=None, N_enh=10, enhancement=False, weight_shat=0.8,weight_y=0.2,  N_epsilon=5, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        self.N_epsilon = N_epsilon
        self.weight_shat = weight_shat
        self.weight_y = weight_y
        ode_cls = ODERegistry.get_by_name(ode)
        self.enhancement = enhancement
        self.N_enh = N_enh
        self.ode = ode_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.ode.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _mse_loss(self, x, x_hat):    
        err = x-x_hat
        losses = torch.square(err.abs())

        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    
    def _loss(self, vectorfield, condVF):    
        if self.loss_type == 'mse':
            err = vectorfield-condVF
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            err = vectorfield-condVF
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x0, y = batch
        
        timesteps = torch.linspace(self.T_rev, self.t_eps, self.N_epsilon, device= y.device)
        
        
        mean, std = self.ode.marginal_prob(x0, torch.ones((y.shape[0],), device=y.device), y)
        z = torch.randn_like(x0)  #
        x_t = y + z * std[:,None,None,None]
        res = 0
        for i in range(len(timesteps)):
            t=timesteps[i]
            vec_t =torch.ones(y.shape[0], device=y.device)*t
            if i != len(timesteps) - 1:
                dt =timesteps[i+1]-t
            else:
                dt =0- timesteps[-1]

            if i != self.N_epsilon-1:                
                with torch.no_grad():
                    x_t = x_t + self.forward(x_t, vec_t,y) * dt
                    
            else:
                #take a Euler step and compute loss
                x_t = x_t + self.forward(x_t, vec_t,y) * dt
                               
                loss = self._loss(x_t, x0)
                break
        
        return loss
    
   
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
    
    def add_para(self, weight_shat):
        self.weight_shat = weight_shat
        self.weight_y = 1-self.weight_shat