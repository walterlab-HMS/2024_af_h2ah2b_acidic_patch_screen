import os, json, requests
import pandas as pd

prev_requests_cache = {}
def get_data_from_url(url, save = True):
    if url in prev_requests_cache:
        return prev_requests_cache[url]
    response = requests.get(url, headers={'Connection':'close'})
    requests.session().close()
    if save:
        prev_requests_cache[url] = response
    return response


def get_pdb_cif_from_pdb_db(pdb_id):
    url = f"https://models.rcsb.org/v1/{pdb_id}/assembly?encoding=cif&copy_all_categories=false&model_nums=1&download=false"
    response =  get_data_from_url(url)
    if response.status_code != 200:
        return None
    pdb_cif = response.text
    return pdb_cif


def get_entry_data(pdb_id):

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    json_data = response.json()
    year = json_data['rcsb_accession_info']['initial_release_date']
    title = json_data['rcsb_primary_citation']['title']
    experimental_method = json_data['exptl'][0]['method']
    resolution_combined = json_data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]

    return {'year':year, 'title':title, 'method':experimental_method, 'resolution':resolution_combined}


pdbs_for_seqs_map = {}
def get_pdb_chains_for_sequence(sequence, datestr = "2099-01-01", comparison='<'):

    search_id = sequence+datestr+comparison
    if search_id in pdbs_for_seqs_map:
        return pdbs_for_seqs_map[search_id]
    print(f"Fetching new chains for sequence:{sequence}")

    data = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "sequence_type": "protein",
                        "value": sequence,
                        "identity_cutoff": 0.7,
                        "evalue_cutoff": 0.1
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "less_or_equal" if comparison == "<" else "greater",
                        "value": datestr,
                        "attribute": "rcsb_accession_info.initial_release_date"
                    }
                },
                 {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "greater_or_equal",
                        "value": 3,
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein"
                    }
                }
            ]},
        "return_type": "polymer_instance",
            "request_options": {
            "return_all_hits": True
        }
    }

    base_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    response = requests.get(base_url, params={"json": json.dumps(data)})
    if response.status_code != 200:
        print("Error getting new chains for sequence")
        pdbs_for_seqs_map[sequence] = []
        return []
    if(len(response.content) < 1):
        print("No chains found for sequence")
        pdbs_for_seqs_map[sequence] = []
        return []
    structure_data = response.json();

    chains = []
    for struct in structure_data["result_set"]:
        comps = struct['identifier'].split('.')
        pdb_id, chain_id = comps
        pdb_id = pdb_id.lower()
        chains.append({"pdb_id":comps[0], "chain_id":chain_id,})

    pdbs_for_seqs_map[search_id] = chains
    return chains


h2a_seq = "APVYMAAVLEYLTAEILELAGNAARDNKKT"
h2b_seq = "TSREIQTAVRLLLPGELAKHAVSEGTKAVT"
afmv3_cutoff_date = "2021-09-30"

h2a_chains = get_pdb_chains_for_sequence(h2a_seq, afmv3_cutoff_date)
h2b_chains = get_pdb_chains_for_sequence(h2b_seq, afmv3_cutoff_date)
h2a_pdbs = set([x['pdb_id'] for x in h2a_chains])
h2b_pdbs = set([x['pdb_id'] for x in h2b_chains])
pre_afmv3_training_h2ah2b_pdbs = list(h2a_pdbs.intersection(h2b_pdbs))
print(f"Found {len(pre_afmv3_training_h2ah2b_pdbs)} PDBs pre-training_cutoff")


h2a_chains = get_pdb_chains_for_sequence(h2a_seq, afmv3_cutoff_date, '>')
h2b_chains = get_pdb_chains_for_sequence(h2b_seq, afmv3_cutoff_date, '>')
h2a_pdbs = set([x['pdb_id'] for x in h2a_chains])
h2b_pdbs = set([x['pdb_id'] for x in h2b_chains])
post_afmv3_training_h2ah2b_pdbs = list(h2a_pdbs.intersection(h2b_pdbs))
print(f"Found {len(post_afmv3_training_h2ah2b_pdbs)} PDBs post-training_cutoff")


all_pdb_ids_found = pre_afmv3_training_h2ah2b_pdbs + post_afmv3_training_h2ah2b_pdbs

if not os.path.isdir('all_pdb_h2ah2b_files'):
    os.mkdir('all_pdb_h2ah2b_files')
    os.mkdir('all_pdb_h2ah2b_files/pre_afmv3_training/')
    os.mkdir('all_pdb_h2ah2b_files/post_afmv3_training/')

pdb_entry_data_rows = []
for ix, pdb_id in enumerate(all_pdb_ids_found):
    print(f"Working on {ix +1}/{len(all_pdb_ids_found)}: {pdb_id}")
    cif_content = get_pdb_cif_from_pdb_db(pdb_id)
    # Define the file path for saving
    period = 'pre' if pdb_id in pre_afmv3_training_h2ah2b_pdbs else 'post'
    entry_data = get_entry_data(pdb_id)
    entry_data['pdb_id'] = pdb_id
    entry_data['period'] = period
    pdb_entry_data_rows.append(entry_data)
    file_path = os.path.join(f"all_pdb_h2ah2b_files/{period}_training", f"{pdb_id}.cif")
    with open(file_path, 'w') as cif_file:
        cif_file.write(cif_content)


entry_df = pd.DataFrame(pdb_entry_data_rows)
entry_df.to_csv('all_pdb_h2ah2b_files/entries.csv', index=None)
