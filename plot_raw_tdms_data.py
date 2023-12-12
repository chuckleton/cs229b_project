import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
print(plt.style.available)
plt.style.use('seaborn-v0_8-poster')

from nptdms import TdmsFile

tdms_file = TdmsFile.open("/etc/test-data-backend/data-files/360777252930/A110-Q096.tdms")

channel_names = ['FTA', 'ATI102', 'PT222', 'PT333']
normalization_means = {
    'FTA': 12.5,
    'ATI102': 3.0,
    'PT222': 520.0,
    'PT333': 520.0,
}
normalization_stds= {
    'FTA': 12.5,
    'ATI102': 3.0,
    'PT222': 25.0,
    'PT333': 25.0,
}

channel_data = {}
for channel_name in channel_names:
    for group in tdms_file.groups():
        group_time = group["Time"][:]
        for channel in group.channels():
            if channel.name == channel_name:
                channel_data[channel_name] = (group_time, channel[:])
                break

# apply normalization
for channel_name in channel_names:
    channel_data[channel_name] = (
        channel_data[channel_name][0],
        (channel_data[channel_name][1] - normalization_means[channel_name]) / normalization_stds[channel_name]
    )

# Trim all channels to the same time range (-1 to 33)
for channel_name in channel_names:
    # trim lower
    start_index = 0
    while channel_data[channel_name][0][start_index] < -1:
        start_index += 1
    channel_data[channel_name] = (
        channel_data[channel_name][0][start_index:],
        channel_data[channel_name][1][start_index:],
    )
    # trim upper
    end_index = -1
    while channel_data[channel_name][0][end_index] > 33:
        end_index -= 1
    channel_data[channel_name] = (
        channel_data[channel_name][0][:end_index],
        channel_data[channel_name][1][:end_index],
    )

# plot
fig, axs = plt.subplots(len(channel_names)-1, 1, figsize=(12, 8), sharex=True)
for i, channel_name in enumerate(channel_names):
    plot_ind = i
    if i == len(channel_names) - 1:
        plot_ind = i-1
    axs[plot_ind].plot(
        channel_data[channel_name][0], channel_data[channel_name][1],
        label=channel_name,
        linewidth=0.5,
    )
    axs[plot_ind].set_title(channel_name)
    if plot_ind == len(channel_names) - 2:
        axs[plot_ind].set_xlabel("Time (s)")
    axs[plot_ind].set_ylabel("Normalized Value")
axs[-1].set_title('PT222 and PT333')
axs[-1].legend()
fig.tight_layout()
fig.savefig("A110-Q096_data.png", dpi=500)

# Plot a zoomed in view from 3 to 4 seconds
fig, axs = plt.subplots(len(channel_names)-2, 1, figsize=(7, 6), sharex=True)
for i, channel_name in enumerate(channel_names):
    plot_ind = i
    if i >= len(channel_names) - 2:
        continue
    axs[plot_ind].plot(
        channel_data[channel_name][0], channel_data[channel_name][1],
        label=channel_name,
        linewidth=1.0,
    )
    axs[plot_ind].set_title(channel_name)
    axs[plot_ind].set_xlim(3, 3.25)
    if plot_ind == len(channel_names) - 2:
        axs[plot_ind].set_xlabel("Time (s)")
    axs[plot_ind].set_ylabel("Normalized Value")
# axs[-1].set_title('PT222 and PT333')
# axs[-1].legend()
fig.tight_layout()
fig.savefig("A110-Q096_data_zoom.png", dpi=500)

