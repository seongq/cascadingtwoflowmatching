import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import CTFSE_MODEL

import torch
torch.set_num_threads(5)
torch.cuda.empty_cache()
def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default="flowmatching")    
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          
          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for VFModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     ode_class = ODERegistry.get_by_name(temp_args.ode)
     parser = pl.Trainer.add_argparse_args(parser)
     CTFSE_MODEL.add_argparse_args(
          parser.add_argument_group("CTFSE_MODEL", description=CTFSE_MODEL.__name__))
     ode_class.add_argparse_args(
          parser.add_argument_group("ODE", description=ode_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)
     dataset = os.path.basename(os.path.normpath(args.base_dir))
     # Initialize logger, trainer, model, datamodule
     model = CTFSE_MODEL(
          backbone=args.backbone, ode=args.ode, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['CTFSE_MODEL']),
               **vars(arg_groups['ODE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )
    
    
     name_save_dir_path = f"CTFSE_dataset_{dataset}_sigma_min_{args.sigma_min}_sigma_max_{args.sigma_max}_T_rev_{args.T_rev}_t_eps_{args.t_eps}"
     logger = WandbLogger(project=f"CTFSE", log_model=True, save_dir="logs", name=name_save_dir_path)
    

     # Set up callbacks for logger

     model_dirpath = f"logs/{name_save_dir_path}_{logger.version}"
     checkpoint_callback_last = ModelCheckpoint(dirpath=model_dirpath,
     save_last=True, filename='{epoch}-last')
     checkpoint_callback_valid_loss = ModelCheckpoint(dirpath=model_dirpath,  save_top_k=1000, monitor="valid_loss", mode="min", filename='{epoch}-{valid_loss:.2f}')
     callbacks = [checkpoint_callback_valid_loss,  checkpoint_callback_last]
     

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          accelerator='gpu', strategy=DDPPlugin(find_unused_parameters=False), gpus=[0,1], auto_select_gpus=False, 
          logger=logger, log_every_n_steps=10, num_sanity_val_steps=1, max_epochs=1000,
          callbacks=callbacks
     )

     # Train model
     trainer.fit(model)

   