import json
import pandas as pd

def read_and_extract_json_data(file_path, fdr = 0.05):

    data_rows = []

    with open(file_path, 'r') as file:
        json_data = json.load(file)

        overrepresentation = json_data.get('overrepresentation', {})
        groups = overrepresentation.get('group', [])

        for g in groups:
            result = g['result']
            if isinstance(result, list):
                for r in result:

                    if r['input_list']['number_in_list'] > 0 and r['input_list']['fdr'] < fdr:
                        data_rows.append({
                            'term':r['term']['label'],
                            'term_id':r['term']['id'],
                            'term_level':r['term']['level'],
                            'fdr':r['input_list']['fdr'],
                            'num_in_list':r['input_list']['number_in_list'],
                            'fold_enrichment':r['input_list']['fold_enrichment'],
                        })
            elif isinstance(result, dict):
                r = result
                if r['input_list']['number_in_list'] > 0 and r['input_list']['fdr'] < fdr:
                    data_rows.append({
                        'term':r['term']['label'],
                        'term_id':r['term']['id'],
                        'term_level':r['term']['level'],
                        'fdr':r['input_list']['fdr'],
                        'num_in_list':r['input_list']['number_in_list'],
                        'fold_enrichment':r['input_list']['fold_enrichment'],
                    })

    df = pd.DataFrame(data_rows)
    new_filename = file_path.split('/').pop()+".csv"
    df.to_csv(new_filename, index=False)

read_and_extract_json_data('MF_AF3_hits_nuclear_bg.json')
