import os,sys, glob, re, math, lzma, gzip
from argparse import ArgumentParser
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd


aa_3c_to_1c = {
    "ALA": 'A',
    "CYS": 'C',
    "ASP": 'D',
    "GLU": 'E',
    "PHE": 'F',
    "GLY": 'G',
    "HIS": 'H',
    "ILE": 'I',
    "LYS": 'K',
    "LEU": 'L',
    "MET": 'M',
    "MSE": 'M',
    "ASN": 'N',
    "PRO": 'P',
    "GLN": 'Q',
    "ARG": 'R',
    "SER": 'S',
    "THR": 'T',
    "VAL": 'V',
    "TRP": 'W',
    "TYR": 'Y',
    "UNK": 'X',
    "SEP": 'S',
    "TPO": 'T',
    "PTR": 'Y',
    "HYP": 'P',
    "MLY": 'K',
    "M3L": 'K',
    "CSO": 'C',
    "PCA": 'E',
    "LLP": 'K',
    "DPR": 'P',
    "BMT": 'T',
    "AIB": 'A',
    "ACL": 'R',
    "DAL": 'A',
    "DAR": 'R',
    "DCY": 'C',
    "DGL": 'E',
    "DGN": 'Q',
    "DHI": 'H',
    "DIL": 'I',
    "DIV": 'V',
    "DLE": 'L',
    "DLY": 'K',
    "DPN": 'F',
    "DSN": 'N',
    "DSP": 'D',
    "DTH": 'T',
    "DTR": 'W',
    "DTY": 'Y',
    "DVA": 'V'
}


representative_vertex_atom = {

    "D":'CG',
    "E":'CD',
    "Q":'CD'
}


def join_csv_files(files:list, output_name:str, sort_col:str = None, sort_ascending:bool = False, headers = None):
    """
        Join multiple CSV files into a single file.

        :param files (list): A list of file paths to CSV files to be joined.
        :param output_name (str): The name of the output file.
        :param sort_col (str, optional): The column header of the final CSV column by which to sort the rows by.
        :param sort_ascending (bool, optional): The sort direction to use when sorting the final output CSV.
        :param headers (list, optional): A list of column names for the output file. If not provided, the column names from the first input file are used.
    """
    if(len(files) < 1):
        return

    all_dfs = []
    for f in files:
        all_dfs.append(pd.read_csv(f))

    combo_df = pd.concat(all_dfs, ignore_index=True)

    if headers is not None:
        combo_df.columns = headers

    if sort_col:
        combo_df.sort_values(by=[sort_col], ascending=sort_ascending, inplace=True)
    combo_df.to_csv(output_name, index=None)


def distribute(lst:list, n_bins:int) -> list:
    """
        Returns a list containg n_bins number of lists that contains the items passed in with the lst argument

        :param lst: list that contains that items to be distributed across n bins
        :param n_bins: number of bins/lists across which to distribute the items of lst
    """ 
    if n_bins < 1:
       raise ValueError('The number of bins must be greater than 0')
    
    #cannot have empty bins so max number of bin is always less than or equal to list length
    n_bins = min(n_bins, len(lst))
    distributed_lists = []
    for i in range(0, n_bins):
        distributed_lists.append([])
    
    for i, item in enumerate(lst):
        distributed_lists[i%n_bins].append(item)

    return distributed_lists


def get_pae_values_from_json_file(json_filename) -> list:
    """
        Returns a list of string values representing the pAE(predicated Aligned Error) values stored in the JSON output

        :param json_filename: string representing the JSON filename from which to extract the PAE values
    """ 

    if not os.path.isfile(json_filename):
        raise ValueError('Non existing PAE file was specified')

    scores_file = None 
    if(json_filename.endswith('.xz')):
        scores_file = lzma.open(json_filename, 'rt')
    elif(json_filename.endswith('.gz')):
        scores_file = gzip.open(json_filename,'rt')
    elif (json_filename.endswith('.json')):
        scores_file = open(json_filename,'rt')
    else:
        raise ValueError('pAE file with invalid extension cannot be analyzed. Only valid JSON files can be analyzed.')

    #read pae file in as text
    file_text = scores_file.read()
    pae_index = file_text.find('"pae":')
    scores_file.close()
    
    #Transform string representing 2d array into a 1d array of strings (each string is 1 pAE value). We save time by not unecessarily converting them to numbers before we use them.
    pae_data = file_text[pae_index + 6:file_text.find(']]', pae_index) + 2].replace('[','').replace(']','').split(',')

    if len(pae_data) != int(math.sqrt(len(pae_data)))**2:
        #all valid pAE files consist of an N x N matrice of scores
        raise ValueError('pAE values could not be parsed from files')
    
    return pae_data


def get_af_model_num(filename) -> int:
    """
        Returns the Alphafold model number from an input filestring as an int

        :param filename: string representing the filename from which to extract the model number
    """ 
    
    if "model_" not in filename: return 0
    model_num = int(re.findall(r'model_\d+', filename)[0].replace("model_", ''))
    return model_num



def dist2(v1, v2) -> float:
    """
        Returns the square of the Euclian distance between 2 vectors carrying 3 values representing positions in the X,Y,Z axis

        :param v1: a vector containing 3 numeric values represening X, Y, and Z coordinates
        :param v2: a vector containing 3 numeric values represening X, Y, and Z coordinates
    """ 
    
    if len(v2) != 3:
        raise ValueError('3D coordinates require 3 values')

    return (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2


def get_data_from_struct_file_line(line, file_type = 'cif'):
    
    if line[0:4] != 'ATOM' and line[0:6] != 'HETATM':
        return None, None, None, None, None, None

    if file_type == 'pdb':
        res_ix = int(line[22:26])
        bfactor = float(line[60:66])
        chain = line[20:22].strip()
        aa_type = line[17:20]
        atom_type = line[13:16].strip()
        coords = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        return chain, res_ix, aa_type, bfactor, atom_type, coords
    elif file_type == 'cif':

        fields = line.split()
        if (fields[0] != 'ATOM' and line[0:6] != 'HETATM') or  len(fields) < 20:
            return None, None, None, None, None, None
        
        atom_type, chain = fields[3],fields[8]
        aa_type = fields[4]
        res_ix = 0
        try:
            res_ix = int(fields[5])
        except:
            print(fields)
            return  None, None, None, None, None, None
        bfactor = float(fields[14])
        coords = np.array([float(fields[10]), float(fields[11]), float(fields[12])])
        return chain, res_ix, aa_type, bfactor, atom_type, coords
    
    return None, None, None, None, None, None


def get_lines_from_struct_file(filename:str) -> list:
    """
        Returns the contents of a structure file as a list of strings

        :param cif_filename: string representing the path of a PDB or CIF file to open and parse (can handle PDB files that have been compressed via GZIP or LZMA)
    """ 

    if not os.path.isfile(filename):
        raise ValueError('Non existing structure file was specified')

    f = None 
    if(filename.endswith('.xz')):
        f = lzma.open(filename, 'rt')
    elif(filename.endswith('.gz')):
        f = gzip.open(filename,'rt')
    elif(filename.endswith('.cif') or filename.endswith('.pdb')):
        f = open(filename,'rt')
    else:
        raise ValueError('Could not parse structure file')
    
    data = f.read()
    f.close()
    return data.splitlines()


def get_amino_acid_sequences_from_structure_file(struct_filepath):

    prev_chain = None
    sequences = []
    active_seq = {}
    chains = []
    file_type = None

    if '.pdb' in struct_filepath:
        file_type = 'pdb'
    elif '.cif' in struct_filepath:
        file_type = 'cif'
    else:
        return None

    for atom_line in get_lines_from_struct_file(struct_filepath):

        chain, res_ix, aa_type, bfactor, atom_type, coords = get_data_from_struct_file_line(atom_line, file_type)
        if chain is None or atom_type != 'CA' or (aa_type not in aa_3c_to_1c):
            continue
        aa_type = aa_3c_to_1c[aa_type]

        if prev_chain != chain:
            if len(active_seq) > 0:
                sequences.append(active_seq)
            active_seq = {}
            prev_chain = chain
            chains.append(chain)

        active_seq[res_ix] = aa_type

    sequences.append(active_seq)
    return dict(zip(chains, sequences))


def get_residue_data_from_structure_file(struct_filepath, residues=None):

    if '.pdb' in struct_filepath:
        file_type = 'pdb'
    elif '.cif' in struct_filepath:
        file_type = 'cif'
    else:
        return None
    
    res_unique_dict = {}

    abs_res_index = 0
    prev_res_ix = None
    for atom_line in get_lines_from_struct_file(struct_filepath):
        chain, res_ix, aa_type, bfactor, atom_type, coords = get_data_from_struct_file_line(atom_line, file_type)
        if chain is None or (aa_type not in aa_3c_to_1c):
            continue

        if prev_res_ix != res_ix:
            abs_res_index += 1
        
        prev_res_ix = res_ix
        res_str = chain + ":" + str(res_ix)
        if residues and res_str not in residues:
            continue

        aa_type = aa_3c_to_1c[aa_type]

        if res_str not in res_unique_dict:
            res_unique_dict[res_str] = {
                'plddt':bfactor,
                'type':aa_type,
                'abs_index':abs_res_index,
                'atoms':{}
            }

        res_unique_dict[res_str]['atoms'][atom_type] = coords


    return res_unique_dict




def get_best_matches(query_seq:str, seqs_to_search:dict, similarity_threshold=0.6):

    query_seq_clean = query_seq.replace("*", "")
    query_seq_len = len(query_seq_clean)
    matches = []
    
    
    for s_ix, seq_dict in enumerate(seqs_to_search):

        seq_aa = "".join(seq_dict.values())
        seq_indices = list(seq_dict.keys())
 
        for i in range(0, len(seq_aa) - query_seq_len):
            
            m_count = 0
            for s1, s2 in zip(query_seq_clean, seq_aa[i:i+query_seq_len]):
                m_count += 1 if s1 == s2 else 0

            if m_count > similarity_threshold*query_seq_len:
                matches.append({
                    'matching_seq_index':s_ix,
                    'matching_in_seq_start_index':seq_indices[i], 
                    'match_count': m_count, 
                    'all_match_in_seq_indices':seq_indices[i:i+query_seq_len]
                })

    if len (matches) < 1:
        return  None
    matches = sorted(matches, key=lambda x: x['match_count'], reverse=True)
    return matches


MAX_TRIANGLE_PERIMETER = 50


def get_closest_vertex(query_vertex, other_vertices):

    min_d2 = 1e6
    closest_v = None
    ix = 0
    for i, v in enumerate(other_vertices):
        d2 = dist2(query_vertex['xyz'], v['xyz'])
        if d2 < min_d2:
            min_d2 = d2
            closest_v = v
            ix = i

    return ix, closest_v


def get_triangles(res_triangle_strs, structure_aa_sequences, all_struct_residue_data):

    res_triangles = []
    structure_chains = list(structure_aa_sequences.keys())

    for triangle_ix, triangle_str in enumerate(res_triangle_strs):

        sub_sequence_regions = triangle_str.split(';')
        vertices_to_consider = []

        for query_seq in sub_sequence_regions:
            seq_matches = get_best_matches(query_seq, structure_aa_sequences.values())
            if seq_matches is None:
                continue
            for seq_match in seq_matches:

                aa_index = 0
                aa_hits = []
                for ix in range(0, len(query_seq) - 1):
                    if query_seq[ix+1] == '*':
                        aa_hits.append(aa_index)
                    if query_seq[ix] != '*':
                        aa_index += 1

                for pos in aa_hits:
                    true_index = seq_match['all_match_in_seq_indices'][pos]
                    chain_name = structure_chains[seq_match['matching_seq_index']]
                    res_code = f"{chain_name}:{true_index}"
                    res_data = all_struct_residue_data[res_code]

                    if res_data['type'] not in representative_vertex_atom:
                        continue

                    if representative_vertex_atom[res_data['type']] not in res_data['atoms']:
                        continue

                    vertices_to_consider.append({
                        'rescode':res_code,
                        'res_data':res_data,
                        'xyz':res_data['atoms'][representative_vertex_atom[res_data['type']]] 
                    })

        if len(vertices_to_consider) < 3:
            continue


        triangle_vertices = []
        active_vertex = vertices_to_consider.pop()
        vertex_group = [active_vertex]
        while active_vertex is not None:

            ix, v =  get_closest_vertex(active_vertex, vertices_to_consider)
            vertex_group.append(v)
            del vertices_to_consider[ix]
            if len(vertex_group) == 3:
                triangle_vertices.append(vertex_group)
    
                if len(vertices_to_consider) > 2:
                    active_vertex = vertices_to_consider.pop()
                    vertex_group = [active_vertex]

                else:
                    active_vertex = None


        for sub_ix, tvs in enumerate(triangle_vertices):

            coordinates = np.array([v['xyz'] for v in tvs])
            distances = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))
            perimeter = np.sum(distances) + np.sqrt(np.sum((coordinates[-1] - coordinates[0])**2))
            if(perimeter > MAX_TRIANGLE_PERIMETER):
                print(f"WARNING: UNUSUALLY LARGE TRIANGLE DETECTED!!!! perimeter = {perimeter}")
                continue

            res_triangles.append({
                "name":f"T{triangle_ix+1}.{sub_ix + 1}",
                "chains":list(set([v['rescode'].split(':')[0] for v in tvs])),
                "vertices":tvs,
                "residues":[v['res_data'] for v in tvs],
                "plddt":round(np.mean(np.array([v['res_data']['plddt'] for v in tvs])), 1),
                "center":np.mean(np.array([v['xyz'] for v in tvs]), axis=0),
                "perimeter":perimeter
            })

    return res_triangles


def analysis_thread_did_finish(a):
    
    print("done")


def get_seq_context(rescode, all_residue_data, num_aas = 10):

    chain, res_ix = rescode.split(':')
    res_ix = int(res_ix)
    num_aas_to_add = num_aas
    seq_context = ''
    scanning_res_ix = res_ix - 1
    new_rescode = f"{chain}:{scanning_res_ix}"
    while num_aas_to_add > 0 and new_rescode in all_residue_data:
        if new_rescode in all_residue_data:
            seq_context += all_residue_data[new_rescode]['type']
        scanning_res_ix -= 1
        num_aas_to_add -= 1
        new_rescode = f"{chain}:{scanning_res_ix}"

    seq_context = seq_context[::-1] + "_"

    num_aas_to_add = num_aas
    scanning_res_ix = res_ix + 1
    new_rescode = f"{chain}:{scanning_res_ix}"
    while num_aas_to_add > 0 and new_rescode in all_residue_data:
        if new_rescode in all_residue_data:
            seq_context += all_residue_data[new_rescode]['type']
        scanning_res_ix += 1
        num_aas_to_add -= 1
        new_rescode = f"{chain}:{scanning_res_ix}"

    return seq_context


def process_files(output_folder, input_folder, cpu_index, struct_filepaths, res_triangle_strs, processing_exp_structs, distance = 6):

    data_lines = []
    d2_cutoff = distance**2

    for ix, filepath in enumerate(struct_filepaths):
        
        print(f"CPU {cpu_index}:: {ix}/{len(struct_filepaths)} ==> {filepath}")

        struct_name = filepath.split("/").pop().replace('.cif', '').replace('.pdb', '')
        model_num = -1
        if not processing_exp_structs:
            struct_name = filepath.split("/").pop().split("_unrelaxed")[0]
            model_num = get_af_model_num(filepath)

        total_aa_length = 0
        pae_data = None

        structure_aa_sequences = get_amino_acid_sequences_from_structure_file(filepath)
        all_residue_data = get_residue_data_from_structure_file(filepath)
        structure_chains = list(structure_aa_sequences.keys())

        triangles = get_triangles(res_triangle_strs, structure_aa_sequences, all_residue_data)
        print(f"Found {len(triangles)} triangles in {struct_name}")
        if len(triangles) < 1:
            print(f"Skipping {struct_name} because no triangles were found")
            continue

        all_relevant_residues = []
        for t in triangles:
            all_relevant_residues += t['vertices']

        triangle_chains = []
        for t in triangles:
            triangle_chains += t['chains'] 

        triangle_chains = list(set(triangle_chains))
        chains_to_search = []
        for chain in structure_chains:
            if chain not in triangle_chains:
                chains_to_search.append(chain)

        anchoring_residues_to_check = []
        for chain in chains_to_search:
            for res_ix, res_type in structure_aa_sequences[chain].items():
                if res_type == 'R':
                    anchoring_residues_to_check.append(f"{chain}:{res_ix}")
        
        all_relevant_residues += anchoring_residues_to_check
        
        for rescode in anchoring_residues_to_check:
            res_data = all_residue_data[rescode]
            if 'CZ' not in res_data['atoms']:
                print(f"PDB:{struct_name} is missing a CZ atom for arginine, {rescode}=>{res_data}")
                continue

            chain, res_ix = rescode.split(':')
            c_coord = res_data['atoms']['CZ']
            min_triangle = None
            min_d2 = 1e5
            for t in triangles:
                new_d2 = dist2(c_coord, t['center'])
                if new_d2 < min_d2:
                    min_d2 = new_d2
                    min_triangle = t

            if min_d2 > d2_cutoff:
                continue


            seq_context = get_seq_context(rescode, all_residue_data)

            if pae_data is None and not processing_exp_structs:
                pae_filename = list(glob.glob(os.path.join(input_folder, f"*{struct_name}*model_{model_num}*.json???")))[0]
                pae_data = get_pae_values_from_json_file(pae_filename)
                total_aa_length = int(math.sqrt(len(pae_data)))

            paes = []
            surrounding_paes = []
            if not processing_exp_structs:
                aa_ix_0 = res_data['abs_index'] - 1
                for res in min_triangle['residues']:
                    aa_ix_1 = res['abs_index'] - 1
                    paes.append(float(pae_data[total_aa_length*aa_ix_0 + aa_ix_1]))
                    paes.append(float(pae_data[total_aa_length*aa_ix_1 + aa_ix_0]))

                neighbor_aa_indices = list(range(aa_ix_0 - 2, aa_ix_0 + 3))
                neighbor_aa_indices.remove(aa_ix_0)
                for aa_ix in neighbor_aa_indices:
                    
                    if aa_ix < 0 or aa_ix > total_aa_length -1:
                        continue
                    for res in min_triangle['residues']:
                        aa_ix_1 = res['abs_index'] - 1
                        surrounding_paes.append(float(pae_data[total_aa_length*aa_ix + aa_ix_1]))
                        surrounding_paes.append(float(pae_data[total_aa_length*aa_ix_1 + aa_ix]))
            else:
                paes = 6*[0]
                surrounding_paes = [0]

            paes = np.array(paes)
            
            data_lines.append([
                        filepath,
                        struct_name,
                        model_num, 
                        chain,
                        res_ix,
                        seq_context,
                        min_triangle['name'],
                        round(min_d2**0.5, 2),
                        min_triangle['plddt'],
                        res_data['plddt'], 
                        np.min(paes),
                        round(np.mean(paes), 1),
                        np.max(paes),
                        round(np.mean(surrounding_paes), 1),
                    ])
            
        
    if len(data_lines) > 0:
        df = pd.DataFrame(data_lines)
        df.columns = ['filepath', 'struct_name', 'model_num', 'chain', 'res_ix', 'seq_context', 'nearest_triangle', 'distance', 'triangle_plddt', 'res_plddt', 'min_pae', 'mean_pae', 'max_pae', 'surrounding_paes']
        df.to_csv(os.path.join(output_folder, f"data_cpu_{cpu_index}.csv"), index=False)



def find_interactors(folder, res_triangle_strs, processing_exp_structs, max_distance):

    struct_files =glob.glob(os.path.join(folder, "*.pdb")) + glob.glob(os.path.join(folder, "*.pdb???")) + glob.glob(os.path.join(folder, "*.cif")) 
    if len(struct_files) < 1:
        print(f"No PDB or CIF structures found in folder specified: {folder}")
        return None
    
    output_folder = folder.rstrip('/').split("/").pop() + "_triangle_analysis"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    num_cpus_to_use = mp.cpu_count()
    num_cpus_to_use = min(num_cpus_to_use,len(struct_files))
    print(f"Splitting analysis job across {num_cpus_to_use} different CPUs")
    pool = mp.Pool(num_cpus_to_use)

    #take the list of complexes and divide it across as many CPUs as we found
    pdb_files_lists = distribute(struct_files, num_cpus_to_use)

    traces = []
    for cpu_index in range(0, num_cpus_to_use):
        #create a new thread to analyze the complexes (1 thread per CPU)
        traces.append(pool.apply_async(process_files, 
                                    args=(output_folder,folder, cpu_index, pdb_files_lists[cpu_index], res_triangle_strs, processing_exp_structs, max_distance), 
                                    callback=analysis_thread_did_finish))

    for t in traces:
        t.get()

    pool.close()
    pool.join()

    data_files = glob.glob(os.path.join(output_folder,'data_cpu_*.csv'))

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"data_{current_datetime}.csv"
    join_csv_files(data_files, os.path.join(output_folder, output_filename))
    for f in data_files:
        os.remove(f)

    return output_filename


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "input",
        default="",
        help="One or more folders with structure files to analyze (.CIF or .PDB supported)",
        nargs='*',) 
    
    parser.add_argument(
        "--triangles",
        default='',
        help="Which sequences to use to define sequence search regions. FORMAT = 'STR1,STR2'",
        type=str,)
    
    parser.add_argument(
        "--expstructs",
        action='store_true',
        help="Set this flag to parse experimental structures. If not set, the script will assume the intent is to analyze AlphaFold multimer structures and will attempt to extract confidence metrics such as pLDDT and PAE.",
        default=False)
    
    parser.add_argument(
        "--distance",
        default=6,
        help="The maximal distance in Angstroms to consider for an anchoring residue",
        type=float,)
    
    args = parser.parse_args()
    res_triangle_strs = args.triangles.split(',')
    if len(res_triangle_strs) < 1:
        sys.exit(f"Please provide at least one triangle sequence string to begin processing")

    for tstr in res_triangle_strs:
        if tstr.count('*') != 3:
            sys.exit(f"Each triangle string must specify 3 residue vertices denoted by a *. This string does not have 3 vertices: {tstr}")

    for folder in args.input:
        if not os.path.isdir(folder):
            print(f"{folder} does not appear to be a valid folder.")
            continue
        find_interactors(folder, res_triangle_strs, args.expstructs, args.distance)
    
