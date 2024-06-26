import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


# Load data
file_path = 'Table_S2_AFM_analysis.xlsx'
data = pd.read_excel(file_path, sheet_name='aggregated_anchors')

print(f"Total number of rows {len(data)}")

# Filter data
in_pdb_data = data[data['in_PDB'] == 1]
not_in_pdb_data = data[data['in_PDB'] == 0]

# Calculate KDE
baseline_kde = gaussian_kde(not_in_pdb_data[['mean_pae', 'distance_angstroms']].T, bw_method=0.5)
not_in_pdb_data['kde_density_estimate'] = baseline_kde(not_in_pdb_data[['mean_pae', 'distance_angstroms']].T)
density_threshold = not_in_pdb_data['kde_density_estimate'].quantile(0.1)
data['kde_density_estimate'] = baseline_kde(data[['mean_pae', 'distance_angstroms']].T)

# Identify outliers
gkde_gated_points = data[data['kde_density_estimate'] < density_threshold]

print(f"{len(gkde_gated_points)}")
print(f"{len(gkde_gated_points)/len(data)}")

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance_angstroms', y='mean_pae', data=data, color='#cc6250', alpha=0.3,edgecolor=None, s=10)
sns.scatterplot(x='distance_angstroms', y='mean_pae', data=gkde_gated_points, color='#228c9e',alpha=0.6,edgecolor=None, s=10)
sns.scatterplot(x='distance_angstroms', y='mean_pae', data=in_pdb_data, color='black',edgecolor='black', s=25,marker='s')

# Add text labels for PDB points
for i, row in in_pdb_data.iterrows():
    label = f"{row['uniprot_entry_name']}_{row['residue_index']}"
    plt.text(row['distance_angstroms'], row['mean_pae'], label, fontsize=8, ha='right')


not_gkde_gated_points = data[data['kde_density_estimate'] >= density_threshold]

plt.xlim([0, 6])
plt.ylim([0, 31])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.xlabel('')
plt.ylabel('')
plt.savefig("step_1.pdf", format="pdf", bbox_inches="tight")
plt.show()


bin_edges = np.histogram_bin_edges(gkde_gated_points['mean_pae'], bins=30)
pae_gated_points = gkde_gated_points[gkde_gated_points['mean_pae'] < 15]

print(f"{len(pae_gated_points)}")
print(f"{len(pae_gated_points)/len(data)}")

plt.figure(figsize=(10, 6))
sns.histplot(gkde_gated_points, x='mean_pae', color='#cc6250', kde=False, label='All Outliers', bins=bin_edges,edgecolor='white')
sns.histplot(pae_gated_points, x='mean_pae', color='#228c9e', kde=False, label='mean_pae < 15', bins=bin_edges, edgecolor='white')

plt.axvline(x=15, color='black', linestyle='--', linewidth=1)
# plt.title('Mean PAE distribution')
plt.xlabel('Mean PAE', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
plt.xlim([0, 30])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.xlabel('')
plt.ylabel('')
plt.savefig("step_2.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Filter the outlier_points to get only those with mean_pae < 15
remaining_pdb_pos_points = pae_gated_points[pae_gated_points['in_PDB'] == 1]

# pae_gated_points['surrounding_paes'],pae_gated_points['_num_models']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='anchor_residue_plddt', y='surrounding_paes', data=pae_gated_points, color='#6d7b9e', alpha=0.2, edgecolor=None, s=10)
sns.scatterplot(x='anchor_residue_plddt', y='surrounding_paes', data=remaining_pdb_pos_points, color='black', edgecolor='black', s=25, marker='s')

med_pdb_gated = remaining_pdb_pos_points[['anchor_residue_plddt', 'surrounding_paes']].median()
std_pdb_gated = remaining_pdb_pos_points[['anchor_residue_plddt', 'surrounding_paes']].std()

# Add ellipse at mean position, 3 standard deviations width and height
ellipse = Ellipse(xy=(med_pdb_gated['anchor_residue_plddt'], med_pdb_gated['surrounding_paes']),
                  width=3*std_pdb_gated['anchor_residue_plddt'], height=3*std_pdb_gated['surrounding_paes'],
                  edgecolor='teal', fc='None', lw=2)
plt.gca().add_patch(ellipse)
plt.axhline(y=0, color='grey', linestyle='--', linewidth=1.5)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

plt.savefig("step_3.pdf", format="pdf", bbox_inches="tight")
plt.show()

ellipse_center = [med_pdb_gated['anchor_residue_plddt'], med_pdb_gated['surrounding_paes']]
ellipse_width = 3*std_pdb_gated['anchor_residue_plddt']
ellipse_height=3*std_pdb_gated['surrounding_paes']

def in_ellipse(row):
    x, y = row['anchor_residue_plddt'], row['surrounding_paes']
    return ((x - ellipse_center[0])**2 / (ellipse_width/2)**2) + ((y - ellipse_center[1])**2 / (ellipse_height/2)**2) <= 1

in_ellipse_points = pae_gated_points[pae_gated_points.apply(in_ellipse, axis=1)]
print(f"Points inside the ellipse: {len(in_ellipse_points)}")
in_ellipse_points.to_csv('final_acidic_patch_hits.csv')
