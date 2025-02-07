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

from flowmse.data_module import SpecsDataModule
from flowmse.model import VFModel_Finetuning, VFModel_Finetuning_SGMSE_CRP
import pdb
import os
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver
from utils import energy_ratios, ensure_dir, print_mean_std

import pdb
torch.set_num_threads(5)
torch.cuda.empty_cache()
if __name__ == '__main__':
    parser = ArgumentParser()
   

    parser.add_argument("--odesolver", type=str,
                        default="euler", help="euler")
    parser.add_argument("--condition", type=str, choices=('noisy', 'mixture', 'enhanced'), required=True)
    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the reverse SDE.")
    parser.add_argument("--reverse_end_point", type=float, default=0.03)
    
    parser.add_argument("--test_dir")
    parser.add_argument("--folder_destination", type=str, help="Name of destination folder.", required=True)    
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--int_list", type=int, nargs='+', help="List of integers")

    # parser.add_argument("--N_mid", type=int, required=True)
    # parser.add_argument("--N", type=int, default=5,required=True, help="Number of reverse steps")    
    parser.add_argument("--weight_shat",type=float, default=0.8)
    parser.add_argument("--starting_state", type=str, choices=('noisy', 'mixture', 'enhanced'), required=True)
    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")
    dataset_name= os.path.basename(os.path.normpath(args.test_dir))
    
    
    checkpoint_file = args.ckpt
    int_list = "_".join(map(str, args.int_list))
    # raise("target_dir 부터 확인해")
    

    # Settings
    sr = 16000
    print(args.int_list)
    odesolver = args.odesolver
    int_list = args.int_list
    
    
    # Load score model
    try:
        model = VFModel_Finetuning.load_from_checkpoint(
            checkpoint_file, base_dir="",
            batch_size=8, num_workers=4, kwargs=dict(gpu=False)
        )
    except:
        model = VFModel_Finetuning_SGMSE_CRP.load_from_checkpoint(
            checkpoint_file, base_dir="", batch_size=8, num_workers=4, kwargs=dict(gpu=False)
        )
        
    target_dir = f"/workspace/test_result/{dataset_name}_{model.mode}_{args.folder_destination}/"
   
    ensure_dir(target_dir + "files/")
    model.weight_shat = args.weight_shat
    reverse_starting_point = args.reverse_starting_point
    reverse_end_point = args.reverse_end_point
    weight_shat = args.weight_shat
    condition = args.condition
    starting_state = args.starting_state
        
    model.ode.T_rev = reverse_starting_point
        
    
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    




    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)

        #pdb.set_trace()        

         
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        Y = Y.cuda()
        with torch.no_grad():
            for i in range(len(int_list)):
                N = int_list[i]
                if N==0:
                    continue
                if i == 0:
                    xt, _ = model.ode.prior_sampling(Y.shape, Y)
                    CONDITION = Y
                else:
                    ENHANCED = xt
                    if starting_state == "enhanced":                        
                        xt, _ = model.ode.prior_sampling(Y.shape,ENHANCED)
                    elif starting_state == "mixture":
                        MIXTURE = 1/2*(Y+ENHANCED)
                        xt, _ = model.ode.prior_sampling(Y.shape, MIXTURE)
                    elif starting_state == "noisy":
                        xt, _ = model.ode.prior_sampling(Y.shape,Y)
                    
                    if condition == "enhanced":
                        CONDITION = ENHANCED
                    elif condition =="mixture":
                        
                        CONDITION = 1/2*(Y+ENHANCED)
                    elif condition == "noisy":
                        CONDITION = Y
                    
                xt = xt.to(Y.device)
                timesteps = torch.linspace(reverse_starting_point, reverse_end_point, N, device=Y.device)
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
        # print("완료")
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
        
        file.write("odesolver: {}\n".format(odesolver))
        file.write("weight_shat: {}\n".format(model.weight_shat))
        file.write("weight_y: {}\n".format(model.weight_y))
        
        file.write("N: {}\n".format(N))
        
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
        file.write("Reverse end point: {}\n".format(reverse_end_point))
        
        file.write("data: {}\n".format(args.test_dir))
        