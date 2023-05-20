from pathlib import Path 

pdbdir = Path('data/dompdb')
outdir = Path('data/preprocessed')
pdb_ids, = glob_wildcards('data/dompdb/{pdb_id}.pdb')
ALL = expand('data/preprocessed/{pdb_id}.pt', pdb_id=pdb_ids)

rule all:
    input: ALL

rule preprocess:
    input:
        pdbdir / '{pdbid}.pdb'
    output:
        outdir / '{pdbid}.pt'
    shell:
        'python scripts/preprocess.py -i {input} -o {output}'