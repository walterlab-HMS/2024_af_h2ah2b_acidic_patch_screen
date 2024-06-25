import requests
import pandas as pd
from argparse import ArgumentParser

seq_cache = {}
def get_protein_sequence(uniprot_id):
  
    if uniprot_id in seq_cache:
        return seq_cache[uniprot_id]

    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        # Extracting the sequence from the FASTA format
        sequence = ''.join(response.text.split('\n')[1:])
        seq_cache[uniprot_id] = sequence
        return sequence
    else:
        print("Error: Unable to fetch the sequence")
        return None
    
entry_data_cache = {}
def get_entry_data(pdb_id):

    if pdb_id in entry_data_cache:
        return entry_data_cache[pdb_id]

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    json_data = response.json()
    release_date = json_data['rcsb_accession_info']['initial_release_date']
    title = json_data['rcsb_primary_citation']['title']
    experimental_method = json_data['exptl'][0]['method']
    resolution_combined = json_data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]

    entry_data_cache[pdb_id] = {'release_date':release_date.split('T')[0], 'title':title, 'method':experimental_method, 'resolution':resolution_combined}
    return entry_data_cache[pdb_id]


def extract_residue_context(sequence, position):
    """
    Extracts the residue at a specific position and +-10 residues around that location,
    padding with '?' if necessary to ensure a fixed length of 21 characters.

    Args:
    sequence (str): The protein sequence.
    position (int): The position of the residue (1-based index).

    Returns:
    str: The specified residue and its surrounding residues.
    """
    # Convert the position to 0-based index
    position -= 1
    
    # Calculate the start and end positions
    start = max(0, position - 10)
    end = min(len(sequence), position + 11)
    
    # Extract the context
    context = sequence[start:end]
    
    # Pad the context with '?' if necessary
    left_padding = '?' * (10 - position) if position < 10 else ''
    right_padding = '?' * (10 - (len(sequence) - position - 1)) if position + 10 >= len(sequence) else ''
    
    return left_padding + context + right_padding


def get_best_match(seq, search):

    seq_len = len(seq)
    search_len = len(search)
    matches = []
 
    for i in range(0, seq_len - search_len):
        
        m_count = 0
        for s1, s2 in zip(search, seq[i:i+search_len]):
            m_count += 1 if s1 == s2 or s1 == '*' else 0

        matches.append([search, i, m_count])

    if len (matches) < 1:
        return  None
    
    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    return matches[0]


map_cache = {}
def get_uniprot_chain_map(pdb_id):

    if pdb_id in map_cache:
        return map_cache[pdb_id]
    
    mapping = {}
    pdb_id = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    response = requests.get(url)
    data = response.json()
    uniprot_data = data[pdb_id]['UniProt']

    for id in uniprot_data:
        name = uniprot_data[id]['name']
        for m in uniprot_data[id]['mappings']:
            chain = m['struct_asym_id']

            p_id = f"{id}:{name}"
            if chain not in mapping:
                mapping[chain] = {p_id:1}
            else:
                mapping[chain][p_id] = 1
    
    final_map = {}
    for chain in mapping:
        final_map[chain] = list(mapping[chain].keys())

    map_cache[pdb_id] = final_map
    return final_map

def process(file_path):
    df = pd.read_csv(file_path)

    if not all(c in df.columns for c in ['pdb_id', 'chain', 'pdb_seq_context']):
      print("Missing required columns! Your CSV file needs to have the columns 'pdb_id', 'chain', 'pdb_seq_context'.")
      return;

    df['uniprot_id'] = ''
    df['uniprot_index'] = ''
    df['uniprot_seq_context'] = ''
    df['protein_name'] = ''
    df['species'] = ''
    df['pdb_title'] = ''
    df['pdb_resolution'] = ''
    df['pdb_release_date'] = ''
    df['pdb_experimental_method'] = ''

    for row_ix, row in df.iterrows():
        pdb_id = row['pdb_id']
        chain = row['chain']
        
        pdb_entry_data = get_entry_data(pdb_id)
        df.at[row_ix, 'pdb_title'] = pdb_entry_data['title']
        df.at[row_ix, 'pdb_resolution'] = pdb_entry_data['resolution']
        df.at[row_ix, 'pdb_release_date'] = pdb_entry_data['release_date']
        df.at[row_ix, 'pdb_experimental_method'] = pdb_entry_data['method']

        res_ix_in_seq = row['pdb_seq_context'].index('_')
        query_seq = row['pdb_seq_context'].replace('_', 'R')

        print(f"working on {row_ix}/{len(df)}")
        chain_uniprot_map = get_uniprot_chain_map(pdb_id)

        active_uniprot_id = None
        if chain not in chain_uniprot_map:
            continue
        uniprot_ids = chain_uniprot_map[chain]
        if len(uniprot_ids) != 1:
            print(f"WARNING: Found {len(uniprot_ids)} uniprot ids associated with CHAIN {chain} in {pdb_id}")
            print(f"Will attempt to find the best match")
        if len(uniprot_ids) < 0:
            print(f"Had to skip CHAIN {chain} in {pdb_id}, no uniprot protein matches found at all")
            continue
        
        best_uniprot_id = uniprot_ids[0]
        best_match = [0,0,0]
        for uid in uniprot_ids:
            active_uniprot_id, protein_name = uid.split(':')
            protein_seq = get_protein_sequence(active_uniprot_id)
            if protein_seq is None:
                continue
            matching_region = get_best_match(protein_seq, query_seq)
            if matching_region[2] > best_match[2]:
                best_uniprot_id = uid
                best_match = matching_region

        if best_match[2] < 1:
            print(f"Had to skip CHAIN {chain} in {pdb_id}, no uniprot protein matches found at all")
            continue

        active_uniprot_id, protein_name = best_uniprot_id.split(':')
        protein_seq = get_protein_sequence(active_uniprot_id)
        uniprot_pos = best_match[1] + res_ix_in_seq + 1

        df.at[row_ix, 'uniprot_id'] = active_uniprot_id
        df.at[row_ix, 'protein_name'] = protein_name
        df.at[row_ix, 'species'] = protein_name.split('_').pop()
        df.at[row_ix, 'uniprot_index'] = uniprot_pos
        df.at[row_ix, 'uniprot_seq_context'] = extract_residue_context(protein_seq, uniprot_pos)


    df.to_csv(file_path.replace(".csv", "_plus_uniprots.csv"), index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "input",
        help="A CSV file that must have columns (pdb_id, chain,pdb_seq_context)",
    )
    args = parser.parse_args()
    process(args.input)
