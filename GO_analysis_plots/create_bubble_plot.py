import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

low_color = "#454866"
high_color = "#ff8a63"
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [low_color, high_color])
norm = mcolors.Normalize(vmin=0, vmax=0.05)


#load CSV file data
file_path = 'MF_AF3_hits_nuclear_bg.json.csv'
df = pd.read_csv(file_path)
df['term_with_id'] = df['term'] + ' (' + df['term_id'] + ')'
df = df.sort_values(by='fold_enrichment', ascending=False)
df = df.head(20)
df = df.sort_values(by='fold_enrichment', ascending=True)


fig, ax = plt.subplots(figsize=(4, 10))
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', alpha=0.7, zorder=1)

scatter = ax.scatter(
    x=df['fold_enrichment'],
    y=df['term_with_id'],
    s=df['num_in_list']*70,  # Scale bubble sizes
    c=df['fdr'],
    cmap=custom_cmap,
    alpha=1,
    edgecolors='w',
    zorder=2,
    norm=norm,
)

ax.set_xlabel('Fold enrichment')
ax.set_ylabel('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, df['fold_enrichment'].max() * 1.05)

ax.set_position([0.1, 0.1, 0.7, 0.8])
# add a gap 
current_xlim = ax.get_xlim()
gap_width = 0.1 * (current_xlim[1] - current_xlim[0])
ax.set_xlim(current_xlim[0] - gap_width, current_xlim[1])

#bold key terms like histone etc
for label in ax.get_yticklabels():
    text = label.get_text().lower()
    if any(word in text for word in ['histone', 'nucleosom', 'chromati']):
        label.set_weight('bold')


# Color bar legend on the side
fdr_legend = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
cbar_ax = fig.add_axes([0.95, 0.2, 0.07, 0.25])  # x, y, width, height in figure coordinate
cbar = plt.colorbar(fdr_legend, cax=cbar_ax)
cbar.ax.set_title('FDR', pad=20)
cbar.outline.set_visible(False)

sizes = [5, 10]
legend_labels = [f'{int(size)}' for size in sizes]
legend_markers = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                  markerfacecolor='black', markersize=np.sqrt(size*70), alpha=1) for size, label in zip(sizes, legend_labels)]

size_legend = ax.legend(handles=legend_markers, title="Count", bbox_to_anchor=(1.16, 0.92), loc='upper left', labelspacing=1.8, handletextpad=1, frameon=False)
ax.add_artist(size_legend)

plt.savefig("molecular_function_GO_analysis.pdf", format="pdf", bbox_inches="tight")
plt.show()
