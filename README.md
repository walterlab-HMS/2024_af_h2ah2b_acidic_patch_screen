# 2024_af_h2ah2b_acidic_patch_screen
Scripts to analyze acidic patch interactions from the PDB and AlphaFold predicted structures.

The accompanying repository for the preprint first released here: https://www.biorxiv.org/content/10.1101/2024.06.26.600687v1

You can interactively view all predicted AlphaFold multimer structures here: https://predictomes.org/view/acidicpatch

The code in this repository has been tested on a Linux machine running Ubuntu 20.04.5 LTS (GNU/Linux 5.15.0-52-generic x86_64). To run the code here please ensure your Python environment has the extra packages installed as specified in the requirements.txt file. 


# Usage information


The scripts are designed to find triangles in structures (experimental or generated by AlphaFold multimer) by searching for amino acid sequences in the structure files, identifying key residues, and then generating triangles by grouping verrtices into triplets. To specify triangles use the --triangles input and supply a string of the form: ```AIRND*EE*LNK;YLTAE*ILELAG```. This tells the script to look for 2 sequences, and find the closest aligning matches and then find the corresponding positions denoted by stars and use those as vertices. For the scripts used in the manuscript we supplied 8 triangle strings to represent the acid patch using 8 triangles.

```
"AIRND*EE*LNK;YLTAE*ILELAG,AIRND*EELNK;YLTAE*ILE*LAG,YLTAE*ILE*LAG;HAVSE*GTKAV,YLTAE*ILELAG;PGE*LAKHAVSE*GTK,YLTAE*ILELAG;AIRNDEE*LNK;PGE*LAKHAV,AAVLE*YLTAEILE*LAG;HAVSE*GTKAV,AAVLE*YLTAEILE*LAG;KVLKQ*VHPDT,SSRAGLQ*FP;AAVLE*YLTAE;KVLKQ*VHPDT"
```

# Examples for running scripts

## Find acidic patch interactors

This is an example command to find acidic patch interactions using the 5 examples files downloaded from the PDB. 4 of 5 have acidic patch interactions, 1 of them does not (7UND). Please note the use of the --expstructs flag which signals to the script that these are experimental (non-AphaFold multimer structures to analyze). Any interactions found will be output in a resulting CSV file. If no inetractions are found no file will be produced.

```
python3 find_acidic_patch_interactions.py acidic_patch_pdb_example_files/ --triangles "AIRND*EE*LNK;YLTAE*ILELAG,AIRND*EELNK;YLTAE*ILE*LAG,YLTAE*ILE*LAG;HAVSE*GTKAV,YLTAE*ILELAG;PGE*LAKHAVSE*GTK,YLTAE*ILELAG;AIRNDEE*LNK;PGE*LAKHAV,AAVLE*YLTAEILE*LAG;HAVSE*GTKAV,AAVLE*YLTAEILE*LAG;KVLKQ*VHPDT,SSRAGLQ*FP;AAVLE*YLTAE;KVLKQ*VHPDT" --expstructs
```

This is an example command to find acidic patch interactions using 4 examples run via AlphaFold-multimer (IL33, HELLS,DPOE1, SHPRH). 3 of 4 have acidic patch interactions, 1 of them does not (DPOE1).

```
python3 find_acidic_patch_interactions.py acidic_patch_afm_example_files/ --triangles "AIRND*EE*LNK;YLTAE*ILELAG,AIRND*EELNK;YLTAE*ILE*LAG,YLTAE*ILE*LAG;HAVSE*GTKAV,YLTAE*ILELAG;PGE*LAKHAVSE*GTK,YLTAE*ILELAG;AIRNDEE*LNK;PGE*LAKHAV,AAVLE*YLTAEILE*LAG;HAVSE*GTKAV,AAVLE*YLTAEILE*LAG;KVLKQ*VHPDT,SSRAGLQ*FP;AAVLE*YLTAE;KVLKQ*VHPDT"
```

## Find and download all H2A/H2B structures in the PDB

```
python3 get_all_pdb_h2ah2b_cif_files.py
```

## Analyze AF3 structures

In this example the script is instructed to analyze all AF3 zip files inside the acidic_patch_af3_example_files folder. It will unzip the file contents, extract residue level contact information, plot PAEs with chain boundaries drawn, and produce a CSV summary file that lists all inter-chain contacts found. By default two residues must meet the following criteria to be consider a valid contact to be reported in the CSV:  The maximum inter-atom distance between the residues must be less than 5 Angstroms. The residue pairs must have at least 1 pair of atoms where both atoms have pLDDTs >= 30. And the mimimum PAE associated with the atom/residue pair must be less than 15 Angstroms.

An optional argument included in this example is the --chain argument. If supplied, the script will only consider and output data associated with contacts ivolving the specified chain. 
```
python3 analyze_af3_zip_files.py acidic_patch_af3_example_files/ --chain A

```
