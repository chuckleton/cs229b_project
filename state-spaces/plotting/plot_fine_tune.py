import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')

labels = [
    'Original',
    'Full',
    'Encoder',
    'L0',
    'L1',
    'L2',
    'L3',
    'Decoder',
    'Encoder + L0',
    'L0 + L1',
    'L1 + L2',
    'L2 + L3',
    'L3 + Decoder',
    'Encoder + Decoder',
]

param_counts = [
    0,
    69.1e3,
    256,
    17.1e3,
    17.1e3,
    17.1e3,
    17.1e3,
    65,
    17.1e3 + 256,
    17.1e3*2,
    17.1e3*2,
    17.1e3*2,
    17.1e3 + 65,
    256 + 65,
]

test_losses = [
    2.959365129470825,
    2.0406463146209717,
    2.055708646774292,
    2.021030902862549,
    1.991052508354187,
    1.977023959159851,
    2.0126757621765137,
    2.5126309394836426,
    2.0131919384002686,
    2.0118813514709473,
    1.9950075149536133,
    1.9555740356445312,
    2.0208382606506348,
    2.065573215484619,
]

# remove 'Decoder' from all lists
labels = labels[0:7] + labels[8:]
param_counts = param_counts[0:7] + param_counts[8:]
test_losses = test_losses[0:7] + test_losses[8:]

# remove original
labels = labels[1:]
param_counts = param_counts[1:]
test_losses = test_losses[1:]

# plot parameter counts vs test loss
# parameter count on log scale

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(param_counts, test_losses, s=150, c='tab:blue', marker='o')
ax.set_xscale('log')
ax.set_xlabel('Tuned Parameter count', fontsize=16)
ax.set_ylabel('Test loss', fontsize=16)
ax.set_title('Fine Tuning', fontsize=22)
# pad below the title

# start x axis at 0
ax.set_xlim(left=0)

for i, label in enumerate(labels):
    # annotate with arrow
    if param_counts[i] == 17.1e3:
        # align right
        ax.annotate(
            label,
            xy=(param_counts[i], test_losses[i]),
            xytext=(-300, 150),
            textcoords='offset pixels',
            arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=10, frac=0.05),
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=14,
        )
    elif param_counts[i] < 1e3:
        ax.annotate(
            label,
            xy=(param_counts[i], test_losses[i]),
            xytext=(200, -100),
            textcoords='offset pixels',
            arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=10, frac=0.05),
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=14,
        )
    elif param_counts[i] > 18e3:
        ax.annotate(
            label,
            xy=(param_counts[i], test_losses[i]),
            xytext=(200, 0),
            textcoords='offset pixels',
            arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=10, frac=0.05),
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=14,
        )
    else:
        ax.annotate(
            label,
            xy=(param_counts[i], test_losses[i]),
            xytext=(150,75),
            textcoords='offset pixels',
            arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=10, frac=0.05),
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=14,
        )

ax.text(
    -0.15, 1.14,
    'Original model:\n0 tuned params, 2.96 loss',
    transform=ax.transAxes,
    verticalalignment='top',
    horizontalalignment='left',
    fontsize=8,
    color='black',
)
ax.text(
    -0.15, 1.08,
    'Decoder only:\n256 tuned params, 2.51 loss',
    transform=ax.transAxes,
    verticalalignment='top',
    horizontalalignment='left',
    fontsize=8,
    color='black',
)

plt.savefig('parameter_count_vs_test_loss.png', dpi=300)
