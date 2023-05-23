from pathlib import Path 

pdbdir = Path('data/gvp_pdb')
outdir = Path('data/gvp_preprocessed')
pdb_ids, = glob_wildcards('data/gvp_pdb/{pdb_id}.pdb')
ALL = expand('data/gvp_preprocessed/{pdb_id}.pt', pdb_id=pdb_ids)

rule all:
    input: ALL

rule preprocess:
    input:
        pdbdir / '{pdbid}.pdb'
    output:
        outdir / '{pdbid}.pt'
    shell:
        'python scripts/preprocess.py -i {input} -o {output}'