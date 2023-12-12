import argparse
from email import message
import os
from pyexpat.errors import messages

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

import hydra
from omegaconf import OmegaConf
from torch.distributions import Categorical
from tqdm.auto import tqdm

from src import utils
from src.dataloaders.audio import mu_law_decode
from src.models.baselines.wavenet import WaveNetModel
from train import SequenceLightningModule

import matplotlib.pyplot as plt
import matplotlib

def test_step(model):
    B, L = 2, 64
    x = torch.ones(B, L, dtype=torch.long).to('cuda')

    # Forward
    batch = (x, None)
    y, _, _ = model(batch) # Forward pass expects a batch which has both x and y (inputs and targets)

    # Step
    model._reset_state(batch, device='cuda')
    ys = []
    for x_ in torch.unbind(x, dim=-1):
        y_ = model.step(x_)
        ys.append(y_)
    ys = torch.stack(ys, dim=1)

    print(torch.norm(y-ys))

    breakpoint()

@torch.inference_mode()
def generate_full(
    model,
    batch,
):
    # Set up the initial state
    model._reset_state(batch, device='cuda')

    x, y, w = model.forward(batch)

    print(f'{x=}, {y=}, {w=}')

    return x


@torch.inference_mode()
def generate(
    model,
    batch,
    tau=1.0,
    l_prefix=0,
    T=None,
    debug=False,
    top_p=1.0,
    benchmark=False,
    return_logprobs=False,
):

    x, _, *_ = batch # (B, L)
    x = x.to('cuda')
    T = x.shape[1] if T is None else T

    # Special logic for WaveNet
    if isinstance(model.model, WaveNetModel) and not benchmark:
        l_prefix += model.model.receptive_field
        T += model.model.receptive_field
        x = F.pad(x, (model.model.receptive_field, 0), value=128)

    # Set up the initial state
    model._reset_state(batch, device='cuda')

    # First sample
    x_t = x[:, 0]
    y_all = []
    logprobs = np.zeros(x.shape[0])
    entropy = np.zeros(x.shape[0])

    if debug:
        y_raw = []

    # Generation loop
    for t in tqdm(range(T)):

        # Step through the model with the current sample
        y_t = model.step(x_t)

        # Handle special loss functions such as ProjectedAdaptiveSoftmax
        # if hasattr(model.loss, "compute_logits"): y_t = model.loss.compute_logits(y_t)

        if debug:
            y_raw.append(y_t.detach().cpu())

        # Output distribution
        # probs = F.softmax(y_t, dim=-1)

        # Optional: nucleus sampling
        # if top_p < 1.0:
        #     sorted_probs = probs.sort(dim=-1, descending=True)
        #     csum_probs = sorted_probs.values.cumsum(dim=-1) > top_p
        #     csum_probs[..., 1:] = csum_probs[..., :-1].clone()
        #     csum_probs[..., 0] = 0
        #     indices_to_remove = torch.zeros_like(csum_probs)
        #     indices_to_remove[torch.arange(sorted_probs.indices.shape[0])[:, None].repeat(1, sorted_probs.indices.shape[1]).flatten(), sorted_probs.indices.flatten()] = csum_probs.flatten()
        #     y_t = y_t + indices_to_remove.int() * (-1e20)

        # Sample from the distribution
        # y_t = Categorical(logits=y_t/tau).sample()

        # Feed back to the model
        if t < l_prefix-1 and t+1 < x.shape[1]:
            x_t = x[:, t+1]
        else:
            x_t = y_t

            # Calculate the log-likelihood
            if return_logprobs:
                probs = probs.squeeze(1)
                if len(y_t.shape) > 1:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t.squeeze(1)]).cpu().numpy()
                else:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t]).cpu().numpy()
                entropy += -(probs * (probs + 1e-6).log()).sum(dim=-1).cpu().numpy()

        # y_all.append(x_t.cpu())
        y_all.append(y_t.cpu())

    y_all = torch.stack(y_all, dim=1) # (batch, length)
    print(y_all.shape)

    # if isinstance(model.model, WaveNetModel) and not benchmark:
    #     y_all = y_all[:, model.model.receptive_field:]


    if not return_logprobs:
        if debug:
            y_raw = torch.stack(y_raw)
            return y_all, y_raw
        print("returning y_all", y_all.shape)
        return y_all
    else:
        assert not debug
        return y_all, logprobs, entropy


@hydra.main(config_path="configs", config_name="generate.yaml")
def main(config: OmegaConf):
    ### See configs/generate.yaml for descriptions of generation flags ###

    # Load train config from existing Hydra experiment
    if config.experiment_path is not None:
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        # config = OmegaConf.merge(config, experiment_config)
        config.model = experiment_config.model
        config.task = experiment_config.task
        config.encoder = experiment_config.encoder
        config.decoder = experiment_config.decoder
        config.dataset = experiment_config.dataset
        config.loader = experiment_config.loader

    # Special override flags
    if not config.load_data:
        OmegaConf.update(config, "train.disable_dataset", True)

    if config.n_batch is None:
        config.n_batch = config.n_samples
    OmegaConf.update(config, "loader.batch_size", config.n_batch)

    # Create the Lightning Module - same as train.py

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    print("Loading model...")
    assert torch.cuda.is_available(), 'Use a GPU for generation.'

    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)

    # Define checkpoint path smartly
    if not config.experiment_path:
        ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)
    else:
        ckpt_path = os.path.join(config.experiment_path, config.checkpoint_path)
    print("Full checkpoint path:", ckpt_path)

    # Load model
    if ckpt_path.endswith('.ckpt'):
        model = SequenceLightningModule.load_from_checkpoint(ckpt_path, config=config)
        model.to('cuda')
    elif ckpt_path.endswith('.pt'):
        model = SequenceLightningModule(config)
        model.to('cuda')

        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location='cuda')
        model.load_state_dict(state_dict)

    # Setup: required for S4 modules in SaShiMi
    for module in model.modules():
        if hasattr(module, '_setup_step'): module._setup_step()
    model.eval()

    if config.load_data:
        # Get the eval dataloaders
        eval_dataloaders = model.val_dataloader()
        dl = eval_dataloaders[0] if config.split == 'val' else eval_dataloaders[1]
    else:
        assert config.l_prefix == 0, 'Only unconditional generation when data is not loaded.'

    # Handle save directory intelligently
    if config.save_dir:
        save_dir = hydra.utils.to_absolute_path(config.save_dir)
    else:
        save_dir = os.path.join(os.getcwd(), "samples/")
    os.makedirs(save_dir, exist_ok=True)

    # Test
    if config.test_model:
        test_step(model)

    # Generate
    assert config.n_samples % config.n_batch == 0, "For convenience, n_samples should be a multiple of n_batch"
    y = []
    logprobs =  []
    for _ in range(config.n_samples // config.n_batch):
        # Construct a batch
        if config.load_data:
            x, y_real, *_ = next(iter(dl))
            x = x.to("cuda")
            batch = (x, None)
        else:
            batch = (torch.zeros(config.n_batch * config.n_reps, 1).to(torch.long) + 128, None)

        # _y = generate(
        #     model, # lightning module (SequenceLightningModule from `train.py`)
        #     batch, # pass data to condition the generation
        #     l_prefix=config.l_prefix, # length of conditioning prefix
        #     T=config.l_sample, # length of generated sequence
        #     top_p=config.top_p, # nucleus sampling: always set to 1.0 for SaShiMi experiments
        #     tau=config.temp, # temperature: always set to 1.0 for SaShiMi experiments
        #     return_logprobs=False, # calc exact likelihoods
        # )
        _y = generate_full(model, batch)
        y.append(_y)
        # logprobs.append(_logprobs)

    # Sort based on likelihoods and save
    y = torch.cat(y, dim=0)
    # logprobs = np.concatenate(logprobs, axis=0)
    # y = y[np.argsort(logprobs.flatten())]

    # Decode quantization
    if config.decode == 'audio':
        print("Saving samples into:", save_dir)
        y = mu_law_decode(y)
        for i, d in enumerate(y):
            filename = f'{save_dir}/unconditional_{config.dataset._name_}_{config.model._name_}_len_{config.l_sample/16000.:.2f}s_gen_{i+1}.wav'
            torchaudio.save(filename, d.unsqueeze(0), 16000)
        np.save(f'{save_dir}/unconditional_{config.dataset._name_}_{config.model._name_}_len_{config.l_sample/16000.:.2f}s_logprobs.npy', logprobs)
    elif config.decode == 'text':
        y = [model.dataset.vocab.get_symbols(_y) for _y in y]
        breakpoint() # Inspect output manually for now
    else: pass

    print("Done!")
    print(f'{y=}')

    x = x.cpu()
    y = y.cpu()

    # save the results
    # data_name = 'test_freq_fine_tune_layer_23_ATP-21_val'
    # data_name = 'test_freq_full_ATP-21_val'
    data_name = 'test_freq_mse_loss_ATP-21_val'
    filename = f'/root/test_data_management/cs229b/saved-results/{data_name}.npz'

    # save the data
    print("Saving data to:", filename)
    np.savez(filename, x=x, y=y, y_real=y_real)
    print("Done saving data to:", filename)

    # Plot the results
    fig, axs = plt.subplots(5,1,figsize=(32,14),sharex=True)
    start_offset = 1000
    total_time = 6
    indices_to_plot = [3,5,8]
    indices_to_plot = [0,1,2,4,7]
    n_lines = len(indices_to_plot)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_lines))
    t = np.linspace(0, total_time, y_real.shape[1])
    fs = 1/(t[1]-t[0])
    for i in range(x.shape[0]):
        # Get the rms error between the generated and real data
        mse = torch.mean((y[i,start_offset:,0] - y_real[i,start_offset:,0])**2)
        # print(f'{mse=}')
        real_integral = torch.sum(y_real[i,start_offset:,0]+1)
        gen_integral = torch.sum(y[i,start_offset:,0]+1)
        integral_error = real_integral - gen_integral
        integral_pct_error = integral_error / real_integral * 100
        print(f'{real_integral=}, {gen_integral=}, {integral_error=}, {integral_pct_error=}%')

    for idx, i in enumerate(indices_to_plot):
    # for i in range(x.shape[0]):
        color = colors[idx]
        axs[-1].plot(t, x[i,:,0], label=f"sample {idx} ATI102", color=color)
        axs[0].plot(t, y[i,:,0], label=f"sample {idx} generated FTA", color=color, linewidth=0.3)
        axs[0].plot(t, y_real[i,:,0], label=f"sample {idx} observed FTA", color=color, linestyle='dashed', alpha=0.5, linewidth=0.3)
        axs[1].plot(t[start_offset:], y_real[i,start_offset:,0]-y[i,start_offset:,0], label=f"sample {idx} FTA error", color=color, linewidth=0.3)
        if idx == 0:
            # Generate a spectrogram of y[i,:,0] and y_real[i,:,0]
            # get the overall min and max of the spectrograms for the real and generated
            # gen_spec, _, _, _ = plt.specgram(y[i,:,0], Fs=2000, NFFT=256, noverlap=128, cmap='viridis')
            # real_spec, _, _, _ = plt.specgram(y_real[i,:,0], Fs=2000, NFFT=256, noverlap=128, cmap='viridis')
            # vmin = min(np.amin(gen_spec), np.amin(real_spec))
            # vmax = max(np.amax(gen_spec), np.amax(real_spec))
            # vmin = 0
            # vmax = 1e-5
            # print(f'{vmin=}, {vmax=}')

            _, _, _, im = axs[2].specgram(
                y[i,:,0], Fs=fs, NFFT=128, noverlap=64, cmap='viridis'
            )
            # Get the vmin and vmax of the generated spectrogram
            vmin, vmax = im.get_clim()
            print(f'{vmin=}, {vmax=}')
            axs[2].set_title("Generated Spectrogram")
            axs[3].specgram(
                y_real[i,:,0], Fs=fs, NFFT=128, noverlap=64, cmap='viridis', vmin=vmin, vmax=vmax
            )
            axs[3].set_title("Real Spectrogram")

    axs[0].legend()
    # axs[1].legend()
    axs[-1].set_xlabel("Time (s)")
    axs[0].set_ylabel("Normalized Values")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[2].set_ylabel("Frequency (Hz)")
    axs[3].set_ylabel("Normalized Values")
    axs[0].set_title("Generated and Real Data Traces")
    axs[-1].set_title("Input ATI102 Signal")
    plt.tight_layout()
    # save the figure
    filename = '/root/test_data_management/cs229b/state-spaces/test_freq_full1.png'
    print("Saving figure to:", filename)
    plt.savefig(filename, dpi=500)

    # fig, axs = plt.subplots(2,1,figsize=(15,12),sharex=True)
    # # Plot of just the real data, adjust normalizations to have 0s instead of 0-mean
    # # x axis is time, same for all, linspace from 0 to 1
    # t = np.linspace(0, 1, y_real.shape[1])
    # ax = axs[0]
    # ax.plot(t, y_real[0,:,0], label="FTA", color="tab:orange")
    # ax.plot(t, x[0,:,0], label="ATI102", color="k")
    # ax.plot(t, x[0,:,1], label="PT222", color="tab:red")
    # ax.plot(t, x[0,:,2], label="PT333", color="tab:green")
    # ax.legend()
    # ax.set_title("Data Traces (Ten Pulses)")
    # ax.set_ylabel("Normalized Values")
    # ax = axs[1]
    # ax.plot(t, y_real[1,:,0], label="FTA", color="tab:orange")
    # ax.plot(t, x[1,:,0], label="ATI102", color="k")
    # ax.plot(t, x[1,:,1], label="PT222", color="tab:red")
    # ax.plot(t, x[1,:,2], label="PT333", color="tab:green")
    # ax.legend()
    # ax.set_title("Data Traces (Single Pulse)")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Normalized Values")
    # # save the figure
    # filename = '/root/test_data_management/cs229b/state-spaces/data_exp.png'
    # print("Saving figure to:", filename)
    # plt.savefig(filename)

    # # Print the model type
    # print("Model type:", type(model.model))
    # for layer in model.model.layers:
    #     print("Layer type:", type(layer.layer.layer.kernel))
    #     # dt, A, B, C, P, Q
    #     dt, A, B, C, P, Q = layer.layer.layer.kernel._get_params()
    #     print(B.shape)
    #     # Print the eigenvalues of A
    #     # print("Eigenvalues of A:", np.linalg.eigvals(A.cpu().detach().numpy()))


if __name__ == "__main__":
    main()
