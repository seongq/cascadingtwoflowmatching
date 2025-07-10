# Speech enhancement based on cascaded two flows
This repository contains the PyTorch implementations for the paper:
* Speech enhancement based on cascaded two flows [1]



<p align="center">  
    <img src="https://seongqjini.com/wp-content/uploads/2025/07/ctfse_interspeech2025_video-ezgif.com-video-to-gif-converter.gif" alt="FlowSE fig1" width="600"/>  
</p>



This repository builds upon previous great works:
* [FlowSE] https://github.com/seongq/flowmse
* [SGMSE] https://github.com/sp-uhh/sgmse  
* [SGMSE-CRP] https://github.com/sp-uhh/sgmse_crp
* [BBED]  https://github.com/sp-uhh/sgmse-bbed
* [StoRM] https://github.com/sp-uhh/storm
## Installation
* Create a new virtual environment with Python 3.10 (we have not tested other Python versions, but they may work).
* Install the package dependencies via ```pip install -r requirements.txt```.
* [**W&B**](https://wandb.ai/) is required.


## Training
Training is done by executing train.py. A minimal running example with default settings (as in our paper [1]) can be run with

```bash
python train.py --base_dir <your_dataset_dir>
```
where `your_dataset_dir` should be a containing subdirectories `train/` and `valid/` (optionally `test/` as well). 

Trained models are saved a director named "logs". 

Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.

To get the training set WSJ0+CHiME3 (H), WSJ0+CHiME3 (L) and WSJ0+Reverb, we refer to https://github.com/sp-uhh/sgmse and https://github.com/sp-uhh/storm.

To see all available training options, run python train.py --help. 

## Evaluation
  To evaluate on a test set, run


  ```bash
  python evaluate_cascading.py --test_dir <your_test_dataset_dir> --folder_destination <your_enh_result_save_dir> --ckpt <path_to_model_checkpoint> --N_second <num_of_time_steps_for_the_second_flow>
  ```


"N_second" is the evaluation number of the numerical integration for the second flow. For the first flow, we set the number of evaluation to be 1. 

`your_test_dataset_dir` should contain a subfolder `test` which contains subdirectories `clean` and `noisy`. `clean` and `noisy` should contain .wav files.
## Citations / References
[1] Seonggyu Lee, Sein Cheong, Sangwook Han, Kihyuk Kim and Jong Won Shin, “*Speech Enhancement based on cascaded two flows*” in Proceedings of Interspeech, Aug. 2025 (accepted).
<!-- 
``` bib
@INPROCEEDINGS{10888274,
  author={Seonggyu Lee and Sein Cheong and Sangwook Han and Jong Won Shin},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={FlowSE: Flow Matching-based Speech Enhancement}, 
  year={2025},
  doi={10.1109/ICASSP49660.2025.10888274}}

``` -->


<!-- Continuous Normalizing Flow (CNF) is a method transforming a simple distribution $p(x)$ to a complex distribution $q(x)$.  

CNF is described by Oridinary Differential Equations (ODEs):  

$$ \frac{d \phi_t(x_0)}{dt} = v(t,\phi_t(x_0)), \phi_0(x_0)=x_0, x_0\sim p(\cdot) $$  

In the above ODE, a function $\phi_t$ called flow is desired such that the stochastic process $x_t=\phi_t(x_0)$ has a marginal distribution $p_t(\cdot)$ such that $p_1(\cdot ) = q(\cdot)$.   
In the above equation, although the condition that $\phi_0(x_0)$ follows $p$ is imposed (inital value problem), by chain rule replacing $t$ with $1-t$, CNF is can be desribed as:  

$$\frac{d\phi_t(x_1)}{dt} = v_t(t,\phi_t(x_1)), \phi_1(x_1)=x_1, x_1 \sim p(\cdot)$$  

It means that it does not matter that the simpled distribution is located at which time point.
Demo page: https://seongqjini.com/speech-enhancement-with-flow-matching-method/ -->