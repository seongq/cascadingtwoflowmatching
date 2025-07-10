import time
import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from torchaudio import load
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd

from flowmse.model import CTFSE_MODEL
import os
from flowmse.util.other import pad_spec
from utils import energy_ratios, ensure_dir, print_mean_std


if __name__ == '__main__':
    parser = ArgumentParser()
   

    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the reverse SDE.")
    parser.add_argument("--reverse_end_point", type=float, default=0.03)
    parser.add_argument("--test_dir")
    parser.add_argument("--folder_destination", type=str, help="Path of destination folder.", required=True)    
    parser.add_argument("--ckpt", type=str, required=True, help='Path to model checkpoint.')
    parser.add_argument("--N_second", type=int, required=True, default=5, help="Number of reverse steps in the second flow")
    
    parser.add_argument("--eval_deb_mod", type=str)
    
    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")
    dataset_name= os.path.basename(os.path.normpath(args.test_dir))
    
    
    eval_deb_mod = args.eval_deb_mod
    
    
    checkpoint_file = args.ckpt

    sr = 16000

    N_second = args.N_second
    assert args.N_second>0
    
    model = CTFSE_MODEL.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=8, num_workers=4, kwargs=dict(gpu=False)
    )
    
    target_dir = f"{args.folder_destination}/"
   
    ensure_dir(target_dir + "files/")


    reverse_starting_point = args.reverse_starting_point
    reverse_end_point = args.reverse_end_point
        
    
    model.eval(no_ema=False)
    model.cuda()
    if eval_deb_mod == "debug":
        noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))[0:10]
    else:
        noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)


         
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        Y = Y.cuda()
        
        xt, _ = model.ode.prior_sampling(Y.shape,Y)
        xt = xt -  model(xt,torch.ones(Y.shape[0], device=Y.device) , Y)
        with torch.no_grad():
            ENHANCED = xt
            xt, _ = model.ode.prior_sampling(Y.shape,ENHANCED)
            CONDITION = 1/2*(Y+ENHANCED)
            for i in range(N_second):
                    
                xt = xt.to(Y.device)
                timesteps = torch.linspace(reverse_starting_point, reverse_end_point, N_second, device=Y.device)
                for i in range(len(timesteps)):
                    t = timesteps[i]
                    if i == len(timesteps)-1:
                        dt = 0-t
                    else:
                        dt = timesteps[i+1]-t
                    vect = torch.ones(Y.shape[0], device=Y.device)*t
                    xt = xt + dt * model(xt, vect, CONDITION)            
                
        
        sample = xt.clone()
        
        
        sample = sample.squeeze()
        
        x_hat = model.to_audio(sample, T_orig)
        y = y * norm_factor
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        
      
        # Convert to numpy
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        write(target_dir + "files/" + filename, x_hat, 16000)

        # Append metrics to data frame
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'wb')
        except: 
            p = float("nan")
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])

    # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("N_second: {}\n".format(N_second))
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
        file.write("Reverse end point: {}\n".format(reverse_end_point))
        file.write("data: {}\n".format(args.test_dir))
        
        