import matplotlib.pyplot as plt
import matplotlib


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 40}
matplotlib.rc('font', **font)

# different markers for each curve
linewidth = 5
markersize = 500

n_mask = [1, 10, 100]

# subplot with two columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# results
gender_sys_accuracy = 0.91
gender_classification = {
    # - key is number of plaintext attacks
    100: [0.643, 0.581, 0.581],
    1000: [0.815, 0.660, 0.594],
    10000: [0.895, 0.777, 0.668],
    100000: [0.925, 0.869, 0.740],
}
smiling_sys_accuracy = 0.884
smiling_classification = {
    100: [0.627, 0.515, 0.508],
    1000: [0.811, 0.581, 0.535],
    10000: [0.871, 0.759, 0.603],
    100000: [0.889, 0.850, 0.701],
}

""" plot gender classification as number of masks increases """

markers = ["o", "s", "v", "D", "X", "P", "d", "p", "h", "H", "8", "x", "1", "2", "3", "4", "s", "p", "P", "*", "h", "H", "+", "x", "D", "d", "|", "_"]

for n_plaintext in gender_classification.keys():
    # plot line with line width
    ax1.plot(n_mask, gender_classification[n_plaintext], linewidth=linewidth)
    # plot points with different markers
    ax1.scatter(n_mask, gender_classification[n_plaintext], marker=markers.pop(0), label=f"{n_plaintext}", s=markersize)
ax1.set_xscale("log")
ax1.set_xlabel("Number of masks")
ax1.set_ylabel("Adversary accuracy")
ax1.grid()

# horizontal line for proposed system
ax1.axhline(y=gender_sys_accuracy, color="k", linestyle="--", label="System\nAccuracy", linewidth=linewidth)
ax1.set_title("Gender")



""" plot smiling classification as number of masks increases """
markers = ["o", "s", "v", "D", "X", "P", "d", "p", "h", "H", "8", "x", "1", "2", "3", "4", "s", "p", "P", "*", "h", "H", "+", "x", "D", "d", "|", "_"]

for n_plaintext in smiling_classification.keys():
    # plot line with line width
    ax2.plot(n_mask, smiling_classification[n_plaintext], linewidth=linewidth)
    # plot points with different markers
    ax2.scatter(n_mask, smiling_classification[n_plaintext], marker=markers.pop(0), label=f"{n_plaintext}", s=markersize)
ax2.set_xscale("log")
ax2.set_xlabel("Number of masks")
ax2.grid()

# horizontal line for proposed system
ax2.axhline(y=smiling_sys_accuracy, color="k", linestyle="--", label="DigiCam\nAccuracy", linewidth=linewidth)

# legend outside to the right
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_title("Smiling")


# save figure
fig.tight_layout()
fig.savefig("privacy_plot.png")
