import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse

import numpy as np

matplotlib.use('Agg')
print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')

##### USER INPUTS #####
data_labels = ['MSE Loss', 'MSE + Spectral Loss', 'MSE + Spectral Loss Fine Tuned']
data_set = 'val'
data_names = ['test_freq_mse_loss_ATP-21', 'test_freq_full_ATP-21', 'test_freq_fine_tune_layer_23_ATP-21']
data_names = [f'{data_name}_{data_set}' for data_name in data_names]

start_offset = 1000
total_time = 5

start_time = 1.4 # start at t=1.4, end at t=3.0 since nothing interesting happens before/after and it's easier to see the shorter time scale
end_time = 3.0

##### END USER INPUTS #####

# model was saved with np.savez(filename, x=x, y=y, y_real=y_real)

filenames = [
    f'/root/test_data_management/cs229b/saved-results/{data_name}.npz'
    for data_name in data_names
]

# Load data
datas = [np.load(filename) for filename in filenames]
xs = [data['x'] for data in datas]
ys = [data['y'] for data in datas]
y_reals = [data['y_real'] for data in datas]
t = np.linspace(0, total_time, y_reals[0].shape[1])

# Get the indices for the start and end times
start_idx = int(start_time * y_reals[0].shape[1] / total_time)
end_idx = int(end_time * y_reals[0].shape[1] / total_time)

# Trim the data
xs = [x[:,start_idx:end_idx,:] for x in xs]
ys = [y[:,start_idx:end_idx,:] for y in ys]
y_reals = [y_real[:,start_idx:end_idx,:] for y_real in y_reals]
t = t[start_idx:end_idx]

layout = [
    ['mse_loss_data', 'mse_loss_spectrogram_real', 'mse_loss_spectrogram_gen'],
    ['mse_loss_error', 'mse_loss_spectrogram_real', 'mse_loss_spectrogram_gen'],
    ['mse_spec_loss_data', 'mse_spec_loss_spectrogram_real', 'mse_spec_loss_spectrogram_gen'],
    ['mse_spec_loss_error', 'mse_spec_loss_spectrogram_real', 'mse_spec_loss_spectrogram_gen'],
    ['fine_tune_data', 'fine_tune_spectrogram_real', 'fine_tune_spectrogram_gen'],
    ['fine_tune_error', 'fine_tune_spectrogram_real', 'fine_tune_spectrogram_gen'],
]

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(30*1.1,9*1.1),
    sharex=True,
    width_ratios=[3, 1, 1],
    height_ratios=[1,0.2,1,0.2,1,0.2],
)
indices_to_plot = [5,7,14]
n_lines = len(indices_to_plot)

colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_lines))
fs = 1/(t[1]-t[0])

prefixes = ['mse_loss', 'mse_spec_loss', 'fine_tune']
for data_idx, (x, y, y_real, prefix) in enumerate(zip(xs, ys, y_reals, prefixes)):
    for idx, i in enumerate(indices_to_plot):
        # for i in range(x.shape[0]):
        color = colors[idx]
        # axs[-1].plot(t, x[i,:,0], label=f"sample {idx} ATI102", color=color)
        axes[prefix+"_data"].plot(t, y[i,:,0], label=f"sample {idx+1}", color=color, linewidth=0.5)
        axes[prefix+"_data"].plot(t, y_real[i,:,0], color=color, linestyle='dashed', alpha=0.75, linewidth=0.7)

        abs_error = np.abs(y_real[i,:,0]-y[i,:,0])
        zeros = np.zeros_like(abs_error)
        axes[prefix+"_error"].fill_between(
            t,
            zeros,
            abs_error,
            color=color,
            alpha=0.5,
            linewidth=0,
        )

        axes[prefix+"_data"].set_xlim([start_time, end_time])
        axes[prefix+"_data"].set_ylim([-3, 3])
        axes[prefix+"_error"].set_xlim([start_time, end_time])
        axes[prefix+"_error"].set_ylim([0, 1.5])
        axes[prefix+"_error"].set_title("Absolute Error")
        # axs[1].plot(t[start_offset:], y_real[i,start_offset:,0]-y[i,start_offset:,0], label=f"sample {idx} FTA error", color=color, linewidth=0.3)
        if idx == 2:
            # Generate a spectrogram of y[i,:,0] and y_real[i,:,0]
            # get the overall min and max of the spectrograms for the real and generated
            # gen_spec, _, _, _ = plt.specgram(y[i,:,0], Fs=2000, NFFT=256, noverlap=128, cmap='viridis')
            # real_spec, _, _, _ = plt.specgram(y_real[i,:,0], Fs=2000, NFFT=256, noverlap=128, cmap='viridis')
            # vmin = min(np.amin(gen_spec), np.amin(real_spec))
            # vmax = max(np.amax(gen_spec), np.amax(real_spec))
            # vmin = 0
            # vmax = 1e-5
            # print(f'{vmin=}, {vmax=}')

            _, _, _, im = axes[prefix+"_spectrogram_real"].specgram(
                y_real[i,:,0], Fs=fs, NFFT=128, noverlap=64, cmap='viridis',
                xextent=(start_time, end_time),
            )
            # Get the vmin and vmax of the real spectrogram
            vmin, vmax = im.get_clim()

            axes[prefix+"_spectrogram_gen"].specgram(
                y[i,:,0], Fs=fs, NFFT=128, noverlap=64, cmap='viridis', vmin=vmin, vmax=vmax,
                xextent=(start_time, end_time),
            )

            # print(f'{vmin=}, {vmax=}')
            # axs[1].set_title("Generated Spectrogram")

            # renumber the xticks to add the start time
            # xticks = axs[2].get_xticks()
            # xticks = xticks + start_time
            # axs[2].set_xticks(xticks)
            # axs[2].set_xticklabels(xticks)
            # axs[1].set_xticks(xticks)
            # axs[1].set_xticks(xticks)
            # axs[2].set_title("Real Spectrogram")

fontsize = 18
# axs[0].legend()
# axs[1].legend()
for ax_label in ['fine_tune_error', 'fine_tune_spectrogram_real', 'fine_tune_spectrogram_gen']:
    axes[ax_label].set_xlabel("Time (s)", fontsize=fontsize)

for i, ax_label in enumerate(['mse_loss_data', 'mse_spec_loss_data', 'fine_tune_data']):
    axes[ax_label].set_ylabel(data_labels[i], fontsize=15)

for i, ax_label in enumerate(['mse_loss_spectrogram_real', 'mse_spec_loss_spectrogram_real', 'fine_tune_spectrogram_real']):
    axes[ax_label].set_ylabel("Frequency (Hz)", fontsize=fontsize)

axes["mse_loss_data"].set_title("Generated (solid) and Real (dashed) Thrust Data Traces", fontsize=24)
axes["mse_loss_spectrogram_real"].set_title("Real Spectrogram", fontsize=24)
axes["mse_loss_spectrogram_gen"].set_title("Generated Spectrogram", fontsize=24)

for ax_label in ['mse_loss', 'mse_spec_loss', 'fine_tune']:
    # Shade the region from x=1.55 to x=1.65 in red
    axes[ax_label+"_data"].axvspan(1.525, 1.6, color='red', alpha=0.1)
    axes[ax_label+"_error"].axvspan(1.525, 1.6, color='red', alpha=0.1)

axes["mse_spec_loss_data"].annotate(
    "Unstable Pulse!",
    xy=(1.6, 2.75),
    xytext=(1.75, 3.25),
    xycoords='data',
    arrowprops=dict(width=2.5, headwidth=10, facecolor='black'),
    fontsize=fontsize,
)
plt.tight_layout()

axes["mse_spec_loss_data"].annotate(
    "",
    xy=(-0.024, 1.2),
    xytext=(-0.024, 1.9),
    xycoords='axes fraction',
    arrowprops=dict(width=2.5, headwidth=10, facecolor='black'),
)
axes["mse_spec_loss_data"].annotate(
    "",
    xytext=(-0.024, -0.15),
    xy=(-0.024, -0.3),
    xycoords='axes fraction',
    arrowprops=dict(width=2.5, headwidth=10, facecolor='black'),
)

# save the figure
filename = '/root/test_data_management/cs229b/model_comparison.png'
print("Saving figure to:", filename)
plt.savefig(filename, dpi=300)