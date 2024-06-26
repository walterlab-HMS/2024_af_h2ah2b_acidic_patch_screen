import os, glob, re, json, gzip, shutil
from datetime import datetime
import numpy as np
import pandas as pd

import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import zipfile


def get_af_model_num(filename) -> int:
    """
        Returns the Alphafold model number from an input filestring as an int
        :param filename: string representing the filename from which to extract the model number
    """ 
    if "model_" not in filename: return 0
    model_num = int(re.findall(r'model_\d+', filename)[0].replace("model_", ''))
    return model_num


def get_distance(a1, a2):
    return np.linalg.norm(a1['xyz'] - a2['xyz'])


def create_pae_img(json_filepath, lines=[], new_png_file_path=None):

    # Determine opener based on file extension
    opener = gzip.open if json_filepath.endswith('.gz') else open
    # Load JSON data
    with opener(json_filepath, 'rt') as f:
        json_data = json.load(f)

    # Extract PAE data and validate
    res_pae_data = np.array(json_data['pae'])
    if res_pae_data.size == 0:
        return None

    # Setup color map
    colors = ["#2d4380", "white", "#f23f48"]  # Light Blue, White, Light Red
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=200)

    # Plot configuration
    fig, ax = plt.subplots(figsize=(1024/300, 1024/300))  
    sns.heatmap(res_pae_data, vmin=0, vmax=30, cmap=cmap, cbar=False, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add lines
    for index in lines:
        ax.axhline(index, color='black', lw=2)
        ax.axvline(index, color='black', lw=2)

    # Remove all margins
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(new_png_file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    return new_png_file_path  # Optionally return the path


def get_data(cif_filepath, json_filepath, max_d=5, min_plddt=30, max_pae = 15, search_chain = None):

    max_d2 = max_d**2
    with open(json_filepath, 'rt') as f:
        json_data = json.load(f)

    res_pae_data = json_data['pae']
    num_residues = len(res_pae_data)
    print(f'Found {num_residues} residues')

    cif_atom_lines = []
    with open(cif_filepath, 'rt') as f:
        # Iterate through each line in the file
        for line in f:
            # Check if the line starts with 'ATOM' or 'HETATM', which are typical for atom entries in CIF files
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Append the line to the list
                cif_atom_lines.append(line.strip())


    structure = {}

    all_atoms = {}
    all_chains = []
    chain_pae_lengths = {}
    all_atom_coords = []

    abs_res_ix = 0

    in_chain_ix = 0

    for atom_line in cif_atom_lines:
        comps = atom_line.split()

        atom_id = int(comps[1])
        res_ix = int(comps[8] if comps[8] != '.' else 1)
        chain = comps[6]

        if (chain not in structure):
            in_chain_ix = 0
            structure[chain] = {}
            all_chains.append(chain)
            all_atom_coords.append([])
            chain_pae_lengths[chain] = 0

        if (res_ix not in structure[chain]):
            structure[chain][res_ix] = {
                'chain':chain,
                'abs_pae_res_indices':[abs_res_ix],
                'in_chain_ix':res_ix,
                'code':comps[5],
                'atoms':{}
            }
            abs_res_ix += 1
            chain_pae_lengths[chain] += 1;
        else: 
            if comps[0] == "HETATM":
                structure[chain][res_ix]['abs_pae_res_indices'].append(abs_res_ix)
                chain_pae_lengths[chain] += 1;
                abs_res_ix += 1

        structure[chain][res_ix]['abs_res_ix'] = abs_res_ix

        coords = [float(comps[10]), float(comps[11]), float(comps[12])]
        
        atom = {
            'id': atom_id,
            'type':comps[3], 
            'xyz':np.array(coords),
            'plddt':float(comps[14]),
            'parent':structure[chain][res_ix]
        }

        structure[chain][res_ix]['atoms'][atom_id] = atom
        all_atoms[f'{chain}_{in_chain_ix}'] = atom;
        all_atom_coords[-1].append(coords)

        in_chain_ix += 1

    
    num_chains = len(all_chains)

    all_atom_contacts = {}
    residue_contacts = {}
    interfaces = {}

    for i in range(0, num_chains):
        chain_1_coords = all_atom_coords[i]
        num_in_c1 = len(chain_1_coords)
        chain1_id = all_chains[i]

        i2_start = i + 1 if num_chains > 1 else i
        for i2 in range(i2_start, num_chains):

            chain2_id = all_chains[i2]
            if search_chain and (chain1_id != search_chain) and (chain2_id != search_chain): continue

            icstr = f'{chain1_id}:{chain2_id}'
            interfaces[icstr] = {
                'avg_plddt':0,
                'num_res_contacts':0,
                'avg_pae':0,
                'num_atom_contacts':0
            }
            residue_contacts[icstr] = {}
            
            atom_c_count = 0
            plddt_sum = 0
            plddt_count = 0
            pae_sum = 0
            pae_count = 0

            chain_2_coords = all_atom_coords[i2]
            num_in_c2 = len(chain_2_coords)

            #construct 2 3D numpy arrays to hold coordinates of atoms
            c1_matrix = np.tile(chain_1_coords, (1, num_in_c2)).reshape(num_in_c1, num_in_c2, 3)
            c2_matrix = np.tile(chain_2_coords, (num_in_c1, 1)).reshape(num_in_c1, num_in_c2, 3)

            #calculate euclidian distance squared (faster) 
            d2s = np.sum((c1_matrix - c2_matrix)**2, axis=2)
            #get residue pairs where amide nitrogens are closer than the initial broad cutoff
            index_pairs = list(zip(*np.where(d2s < max_d2)))

            for a1_ix, a2_ix in index_pairs:

                #check for cases when inside the same chain and you don't want to report adjacent residues
                if chain1_id == chain2_id and abs(a2_ix - a1_ix) < 3:continue

                atom_1 = all_atoms[f'{chain1_id}_{a1_ix}']
                atom_2 = all_atoms[f'{chain2_id}_{a2_ix}']
                if atom_1['plddt'] < min_plddt or atom_2['plddt'] < min_plddt: continue

                r1_ix = atom_1['parent']['in_chain_ix']
                r2_ix = atom_2['parent']['in_chain_ix']

                pae_data = []
                for ix1 in atom_1['parent']['abs_pae_res_indices']:
                    for ix2 in atom_2['parent']['abs_pae_res_indices']:

                        pae_1 = res_pae_data[ix1][ix2]
                        pae_2 = res_pae_data[ix2][ix1]
                        pae_data.append(pae_1)
                        pae_data.append(pae_2)
                        

                if min(pae_data) > max_pae: continue

                contact_id = f'{r1_ix}&{r2_ix}'

                if contact_id not in residue_contacts[icstr]:
                    residue_contacts[icstr][contact_id] = {
                        'codes':[atom_1['parent']['code'], atom_2['parent']['code']],
                        'paes':pae_data,
                        'plddts':[[], []],
                        'min_distance':1e5,
                        'num_atom_pairs':0,
                        'closest_atoms':[]
                    }
                    pae_sum += sum(pae_data)
                    pae_count += len(pae_data)
                
                plddt_sum += atom_1['plddt'] + atom_2['plddt']
                plddt_count += 2
                atom_c_count += 1

                atom_contact_id = f'{atom_1["id"]}-{atom_2["id"]}'
                all_atom_contacts[atom_contact_id] = 1;
                residue_contacts[icstr][contact_id]['plddts'][0].append(atom_1['plddt'])
                residue_contacts[icstr][contact_id]['plddts'][1].append(atom_2['plddt'])
                
                d = get_distance(atom_1, atom_2)
                if d < residue_contacts[icstr][contact_id]['min_distance']:

                    residue_contacts[icstr][contact_id]['min_distance'] = d
                    residue_contacts[icstr][contact_id]['closest_atoms'] = [atom_1['type'], atom_2['type']]

                residue_contacts[icstr][contact_id]['num_atom_pairs'] += 1
                
            if atom_c_count > 0:
                interfaces[icstr] = {
                    'avg_plddt':round(plddt_sum/plddt_count, 2),
                    'num_res_contacts':pae_count/2,
                    'num_atom_contacts':atom_c_count,
                    'avg_pae':round(pae_sum/pae_count, 2)
                }

    return chain_pae_lengths, interfaces, residue_contacts, all_atom_contacts




def process_complex(input_folder_path, result_folder_path, complex_name, complex_files, max_d, min_plddt, max_pae, chain):

    num_models = len(complex_files.keys())

    atom_contacts_across_models = {}
    all_interfaces = {}
    all_residue_contacts = {}
    pae_img_urls = {}

    chains = None

    for model_num, model_datafiles in complex_files.items():
        chain_pae_lengths, interfaces, res_contacts, all_atom_contacts = get_data(model_datafiles['cif'], model_datafiles['json'], max_d, min_plddt, max_pae, chain)
        chains = list(chain_pae_lengths.keys())
        chain_x_positions = []
        active_chain_x = 1
        for c in chains:
            active_chain_x += chain_pae_lengths[c]
            chain_x_positions.append(active_chain_x)


        if len(chain_x_positions) > 1: chain_x_positions.pop()

        new_pae_img_url = os.path.join(result_folder_path, f'{complex_name}_pae_model_{model_num}.png')
        create_pae_img(model_datafiles['json'], chain_x_positions, new_pae_img_url)
        pae_img_urls[model_num] = new_pae_img_url

        summary_json_data = None
        if model_datafiles['summary'] is not None:
            with open(model_datafiles['summary'], 'rt') as f:
                summary_json_data = json.load(f)

            iptm_matrix = summary_json_data['chain_pair_iptm']

        all_interfaces[model_num] = interfaces
        all_residue_contacts[model_num] = res_contacts
        
        for cid in all_atom_contacts:
            if cid not in atom_contacts_across_models:
                atom_contacts_across_models[cid] = 0

            atom_contacts_across_models[cid] += 1
    
    c = 0
    s = 0
    for cid in atom_contacts_across_models:
        s += atom_contacts_across_models[cid]
        c += 1

    avg_n_models = 0
    if c > 0:
        avg_n_models = round(s/c, 2)

    unique_res_contacts = {}

    for model_num, res_contacts in all_residue_contacts.items():
             for icstr, contacts in res_contacts.items():
                    for cid, contact in contacts.items():
                        contact_uid = "|" + icstr + "|" + cid + "|"
                        if contact_uid not in unique_res_contacts:
                            unique_res_contacts[contact_uid] = []
                        unique_res_contacts[contact_uid].append(str(model_num))


    interface_model_counts = {}

    observed_contacts = {}

    for model_num, res_contacts in all_residue_contacts.items():
             for icstr, contacts in res_contacts.items():
                    
                    if icstr not in interface_model_counts:
                        interface_model_counts[icstr] = {'sum':0, 'count':0, 'avg_n_models':0}

                    for cid, contact in contacts.items():
                        contact_uid = "|" + icstr + "|" + cid + "|"
                        models = unique_res_contacts[contact_uid]
                        models.sort()
                        contact['models'] = models
                        
                        if contact_uid not in observed_contacts:
                            observed_contacts[contact_uid] = True
                            interface_model_counts[icstr]['sum'] += len(models)
                            interface_model_counts[icstr]['count'] += 1

    for icstr in interface_model_counts:
        if(interface_model_counts[icstr]['count'] > 0):
            interface_model_counts[icstr]['avg_n_models'] = round(interface_model_counts[icstr]['sum']/interface_model_counts[icstr]['count'], 2)

    return {
        'complex_name':complex_name,
        'pae_img_urls':pae_img_urls,
        'chains':chains,
        'all_interfaces':all_interfaces,
        'avg_n_models':avg_n_models,
        'num_models_run':num_models,
        'all_residue_contacts':all_residue_contacts,
        'interface_model_counts':interface_model_counts
    }


def analyze_folder(folder_path, settings):

    result_folder = folder_path.rstrip('/') + '_AF3_ANALYSIS'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    complexes = {}
    all_cif_filepaths = glob.glob(os.path.join(folder_path, "*.cif"))
    for cif_path in all_cif_filepaths:

        cif_filename = cif_path.split('/').pop()
        complex_name = cif_filename.split('_model_')[0]
        model_num = get_af_model_num(cif_filename)

        json_path = cif_path.replace('_model_', '_full_data_').replace('.cif', '.json')
        if not os.path.exists(json_path): continue

        summary_json_path = cif_path.replace('_model_', '_summary_confidences_').replace('.cif', '.json')

        if complex_name not in complexes:
            complexes[complex_name] = {}

        if model_num not in complexes[complex_name]:
            complexes[complex_name][model_num] = {'cif':None, 'json':None, 'summary':None}

        complexes[complex_name][model_num]['cif'] = cif_path
        complexes[complex_name][model_num]['json'] = json_path

        if os.path.exists(summary_json_path): 
            complexes[complex_name][model_num]['summary'] = summary_json_path 


    all_contacts = []
    summary_data = []
    all_pae_img_urls = []

    for complex_name, complex_files in complexes.items():
        data = process_complex(folder_path, result_folder, complex_name, complex_files, settings['max_d'], settings['min_plddt'], settings['max_pae'], settings['chain'])
        summary_data.append({
            'complex_name':complex_name,
            'interface_model_counts':data['interface_model_counts']
        })

        for model_num, url in data['pae_img_urls'].items():
            all_pae_img_urls.append(url)

        for model_num, res_contacts in data['all_residue_contacts'].items():

             for icstr, contacts in res_contacts.items():
                    c1, c2 = icstr.split(':')
                    for cid, contact in contacts.items():

                        r1_ix, r2_ix = cid.split('&')
                        all_contacts.append({
                            'complex_name':complex_name,
                            'model_num':model_num,
                            'chain_1':c1,
                            'chain_2':c2,
                            'res_1_ix':r1_ix,
                            'res_1_code':contact['codes'][0],
                            'res_2_ix':r2_ix,
                            'res_2_code':contact['codes'][1],
                            'res_1_atom_plddt_avg':round(np.mean(np.array(contact['plddts'][0])), 1),
                            'res_2_atom_plddt_avg':round(np.mean(np.array(contact['plddts'][1])), 1),
                            'min_inter_res_pae': min(contact['paes']),
                            'min_distance': round(contact['min_distance'], 1),
                            'num_atom_pairs':contact['num_atom_pairs'],
                            'res_1_closest_atom':contact['closest_atoms'][0],
                            'res_2_closest_atom':contact['closest_atoms'][1],
                            'model_count':len(contact['models']),
                            'found_in_models':";".join(contact['models'])
                        })


    time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    contacts_csv_path = None
    if len(all_contacts) > 0:
        contacts_csv_path = os.path.join(result_folder, f"af3_analysis_res_interactions_{time_string}.csv")
        contact_df = pd.DataFrame(all_contacts)
        contact_df.sort_values(by = ["complex_name", "model_num", "chain_1", "res_1_ix", "chain_2", "res_2_ix"], inplace=True)
        contact_df.to_csv(contacts_csv_path, index=None)

    summary_csv_path = None
    if len(summary_data) > 0:
        summary_csv_path = os.path.join(result_folder, f"af3_summary_{time_string}.csv")

        export_data = []
        for d in summary_data:
            cname = d['complex_name']
            for icstr, if_data in d['interface_model_counts'].items():
                export_data.append({
                    'complex_name':cname,
                    'interface':icstr,
                    'avg_n_models':if_data['avg_n_models'],
                    'n_unique_res_contacts':if_data['count']
                })

        summary_df = pd.DataFrame(export_data)
        summary_df.sort_values(by = ["complex_name", "interface"])
        summary_df.to_csv(summary_csv_path, index=None)


    pae_img_folder = result_folder + '/pae_images/'
    os.mkdir(pae_img_folder)
    for url in all_pae_img_urls:
        filename = url.split('/').pop()
        shutil.move(url, os.path.join(pae_img_folder, filename))


def main(folder, settings):

    print(folder)

    if not os.path.exists(folder):
        print(f"Could not find folder {folder}, skipping analysis")
        return False
    
    new_output_foldername = folder.rstrip('/') + "_UNZIPPED"
    new_output_foldername = new_output_foldername.replace("-", "_")
    new_output_folderpath = os.path.join(new_output_foldername)

    folder_already_existed = False

    if not os.path.exists(new_output_folderpath):
        os.mkdir(new_output_folderpath)
    else:
        folder_already_existed = True

    analysis_success = False
    try:

        if not folder_already_existed:
            print("Unzipping contents")
            zip_files = glob.glob(folder + '/*.zip')
            print(f"Found {len(zip_files)} ZIP files in the specified directory.")
            for zip_file_path in zip_files:
                print(f"Working on {zip_file_path}")
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:

                    for item in zip_ref.infolist():

                        if(item.filename.startswith('.') or '..' in item.filename):
                            raise Exception("Invalid filename")
                        if(not bool(re.match(r'^[\w.-]+$', item.filename))):
                            raise Exception("Invalid filename")

                        if (item.filename.endswith('.cif') or re.search(r'data_\d+\.json', item.filename) != None or re.search(r'summary_confidences_\d+\.json$', item.filename) != None):
                            zip_ref.extract(item, new_output_folderpath)

        analyze_folder(new_output_folderpath, settings)
        analysis_success = True

    except Exception as e:
        print(f"Analysis error: {e}")
        analysis_success = False

    if os.path.exists(new_output_folderpath):
        shutil.rmtree(new_output_folderpath)

    return analysis_success


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a folder of AlphaFold 3 server ZIP files to get residue level interaction data and PAE image files.")
    parser.add_argument('folder', type=str, help='The path to the folder to be processed. The folder should contain one or more ZIP files directly output from the AlphaFold 3 server.')
    parser.add_argument('--chain', type=str, help='A single letter representing the chain for which to extract contacts for.', default=None)

    args = parser.parse_args()
    main(args.folder, {'max_d':5,'min_plddt':30, 'max_pae':15, 'chain':args.chain})
