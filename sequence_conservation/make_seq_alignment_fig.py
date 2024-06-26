import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
species_name_map = {
  'hs':'H. sapiens',
  'rn':'R. norvegicus',
  'mm':'M. musculus',
  'xt':'X. tropicalis',
  'dr':'D. rerio',
  'dm':'D. melanogaster',
  'ce':'C. elegans',
  'sc':'S. cerevisiae',
  'sp':'S. pombe',
}

color_map = {
    0: '#ededed', 1: '#f58e49', 2: '#a4e0d4', 3: '#5975ba',
    4: '#fff9a8', 5: '#5975ba', 6: '#a4e0d4', 7: '#ededed',
    8: '#f58e49', 9: '#ededed', 10: '#ededed', 11: '#f58e49',
    12: '#ededed', 13: '#ededed', 14: '#ededed', 15: '#a4e0d4',
    16: '#a4e0d4', 17: '#ededed', 18: '#a4e0d4', 19: '#ededed',
    20: '#ffffff'
}


def aa_number(aa_code):
  return amino_acids.index(aa_code)


def get_alignment_block(seqs, id, region = None):

    first_seq = seqs[list(seqs.keys())[0]]
    target_seq_name = [k for k in seqs.keys() if id in k].pop()
    target_seq = seqs[target_seq_name].strip()

    index_map = {}
    total_alignment_length = len(first_seq)
    in_seq_index = 0
    for abs_index in range(0, total_alignment_length):

      if(target_seq[abs_index] != '-'):
        in_seq_index += 1
        index_map[in_seq_index] = abs_index

    if region is None:
        region = {'start':1, 'end':max(index_map.keys())}

    else:
      region['start'] = max(1, region['start'])
      region['end'] = min(max(index_map.keys()), region['end'])

    seq_block = []
    percent_identical = []
    for seq_name, seq in seqs.items():

      seq_block_row = []
      num_identical_to_target = 0
      for seq_index in range(region['start'],region['end'] + 1):
          abs_index = index_map[seq_index]
          aa = aa_number(seq[abs_index])
          num_identical_to_target += 1 if aa == aa_number(target_seq[abs_index]) else 0
          seq_block_row.append(aa)
      percent_identical.append(round(100*num_identical_to_target/(region['end'] - region['start'] + 1), 1))
      seq_block.append(seq_block_row)
    return seq_block, percent_identical


def calculate_conservation(block):
    conservation_scores = []
    num_seqs = len(block)
    for col in range(len(block[0])):
        col_data = [row[col] for row in block]
        most_common = max(set(col_data), key=col_data.count)
        conservation_scores.append(col_data.count(most_common) / num_seqs)
    return conservation_scores


def display_alignment_block(block, sequence_names, position_offset = 1, highlight_columns = [], identity_percentages = []):

  numpy_data = np.array(block)
  num_rows, num_cols = numpy_data.shape
  conservation_scores = calculate_conservation(block)
  # Define your color mapping (0 to 20)
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10 + round(num_cols/10), 6),  gridspec_kw={'height_ratios': [15, 4]}, tight_layout=True)

  # MSA plot
  cmap = mcolors.ListedColormap([color_map[i] for i in range(21)])
  cax = ax1.matshow(numpy_data, cmap=cmap)

  show_identity_percentages = len(identity_percentages) == num_rows
  for i in range(num_rows):
      for j in range(num_cols):
          text = ax1.text(j, i, amino_acids[numpy_data[i, j]],
                          ha="center", va="center", color="black", fontsize=18)
      ax1.axhline(y=i + 0.5, color='white', linewidth=2)

      font_weight = 'bold' if 'sapiens' in sequence_names[i] else 'normal'
      ax1.text(-1, i, sequence_names[i], ha='right', va='center', color='black', fontsize=16, weight=font_weight)
      if show_identity_percentages and 'sapiens' not in sequence_names[i]:
        ax1.text(num_cols + 0.15, i, f"{identity_percentages[i]} %", ha='left', va='center', color='black', fontsize=16, weight=font_weight)

  ax1.set_xticks(np.arange(0, num_cols, 5))  # Set ticks every 5 positions
  ax1.set_xticklabels([str(position_offset + i) for i in range(0, num_cols, 5)], rotation=0)  # Horizontal labels
  ax1.tick_params(axis='both', which='both', length=0, labelsize=16)
  ax1.set_yticks([])
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.spines['bottom'].set_visible(False)
  ax1.spines['left'].set_visible(False)

  for c in highlight_columns:
      c -= position_offset
      highlight_rect = patches.Rectangle((c-0.5, -0.5), 1, num_rows, edgecolor='black', facecolor='none', lw=1,zorder=10)
      ax1.add_patch(highlight_rect)


  default_bar_color = '#D3D3D3'
  highlight_bar_color = 'black'  # Define the highlight color

  for pos in range(len(conservation_scores)):
      bar_color = highlight_bar_color if (position_offset + pos) in highlight_columns else default_bar_color
      ax2.bar(pos, conservation_scores[pos], color=bar_color, edgecolor='white')
  ax2.set_ylim(0, 1)
  ax2.set_xlim(-0.5, num_cols - 0.5)  # Aligning the bars with the matrix
  ax2.set_ylabel('')
  ax2.set_xlabel('')
  ax2.tick_params(axis='both', labelsize=16)
  ax2.set_xticks(np.arange(0, num_cols, 5))  # Set ticks every 5 positions
  ax2.set_xticklabels([str(position_offset + i) for i in range(0, num_cols, 5)], rotation=0)  # Horizontal labels
  ax2.spines['top'].set_visible(False)
  ax2.spines['right'].set_visible(False)

  plt.savefig("msa.pdf", format="pdf", bbox_inches="tight")
  plt.show()


def get_sequences_from_file(file_path):

  raw_alignment_txt = None
  with open(file_path) as f:
    raw_alignment_txt = f.read()

  sequences = {}
  for lines in raw_alignment_txt.split('\n'):
    comps = lines.split(' ')
    if len(comps) != 2:
      continue

    name = comps[0]
    seq = comps[1]
    if name in sequences:
      sequences[name] += seq
    else:
      sequences[name] = seq

  seq_ids = list(sequences.keys())
  for id in seq_ids:
      if id.split('|')[0] not in species_name_map.keys():
        del sequences[id]

  seq_ids = list(sequences.keys())
  new_sequences = {}
  for species in species_name_map.keys():
      for id in seq_ids:
        if species in id:
          new_sequences[id] = sequences[id]

  return new_sequences




sequences = get_sequences_from_file('/content/SHPRH_aligned_seqs.fasta.results')
central_position = 793
highlight_positons = [793, 796]
window_size = 12

block, in_block_percentages = get_alignment_block(sequences, 'hs', {'start':central_position-window_size, 'end':central_position+window_size})
seq_species = [species_name_map[id.split('|')[0]] for id in sequences.keys()]
display_alignment_block(block, seq_species, central_position-window_size, highlight_positons, in_block_percentages)
