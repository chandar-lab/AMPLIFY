# AMPLIFY Data Pipeline

## Validation Sets

### Reference Proteome Selection (UniProt)

#### Compare EBI and UniProt Reference Proteomes

The list of reference proteomes from UniProt is extensive (see [UniProt README](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/README)) and includes many unassembled WGS, mitochondria-only, and plasmid sequences. Therefore, we used a more refined list of reference proteomes from EBI (see [EBI Reference Proteomes](https://www.ebi.ac.uk/reference_proteomes/)).

Alternatively, you could search UniProt proteomes with the following query: `proteome_type:1` AND `busco:[60 TO 100]`. However, since we used the EBI reference proteomes, we compared them with UniProt to assign BUSCO scores.

```python
# Load spreadsheets containing proteome information from UniProt and EBI
df_proteomes = pd.read_excel('proteomes_proteome_type_1_AND_busco_60_2023_10_31.xlsx',)
df_ebi_ref_proteomes =  pd.read_excel('ebi-reference-proteomes-clean.xlsx')

# Extract BUSCO scores from the UniProt data
def split_string(x):
    pattern = re.search(r'C:(.*?)%\[', x).group(1)
    return float(pattern)

df_proteomes['busco_c'] = df_proteomes.apply(lambda x: split_string(x['BUSCO']), axis=1)

# Find the intersection between EBI and UniProt reference proteomes
ebi_ref_species_list = df_ebi_ref_proteomes['Taxon mnemonic'].to_list()
uniprot_ref_species_list = df_proteomes['Taxon mnemonic'].to_list()

# Filter UniProt data to include only EBI species and save to file
dfebiproteomes_busco = df_proteomes[df_proteomes['Taxon mnemonic'].isin(ebi_ref_species_list)]
dfebiproteomes_busco.to_excel('ebirefproteomes-with-busco-info.xlsx')

# Create a validation set of reference proteomes with BUSCO scores
species_list = ['ARATH', 'BACSU', 'BOVIN', 'CAEEL', 'CALJA', 'DANRE', 'DROME', 'ECOLI', 'HUMAN', 'MAIZE', 'MOUSE', 'MYCTU', 'ORYSJ', 'PANTR', 'PIG', 'RAT', 'SCHPO', 'THEMA', 'THET8', 'XENLA', 'XENTR', 'YEAST', 'THEKO', 'HALSA', 'METJA', 'DICDI', 'PLAF7', 'THAPS', 'RHOBA', 'USTMA', 'CERPU', 'ASPFU']

df_validation_proteomes = df_proteomes[df_proteomes['Taxon mnemonic'].isin(species_list)]
df_validation_proteomes.to_excel('validation_proteome_set.xlsx')
valset_proteomes_list = df_validation_proteomes['Proteome Id'].to_list()
```

#### Download Selected Reference Proteomes from UniProt

```python
import requests

ids = ['UP000000437','UP000000532','UP000000536','UP000000554','UP000000561',
	    'UP000000589','UP000000625','UP000000803','UP000000805','UP000001025',
		'UP000001449','UP000001450','UP000001570','UP000001584','UP000001940',
		'UP000002195','UP000002277','UP000002311','UP000002485','UP000002494',
		'UP000005640','UP000006548','UP000007305','UP000008143','UP000008183',
		'UP000008225','UP000008227','UP000009136','UP000059680','UP000186698',
		'UP000822688','UP000002530']

def fetch_savetofile(url, id):
	with requests.get(url, stream=True) as request:
		request.raise_for_status()
		with open(f'{id}.fasta.gz', 'wb') as f:
			for chunk in request.iter_content(chunk_size=2**20):
				f.write(chunk)

for id in ids:
	url = f'https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28proteome%3A{id}%29'
	print(url)
	fetch_savetofile(url, id)
```

### Convert FASTA Files to CSV Format

The downloaded files are in `*.fasta.gz` format. The relevant information, such as protein existence, species, and gene name, is stored in a single header line. For example:

```
">sp|A0A0G2JMI3|HV692_HUMAN Immunoglobulin heavy variable 1-69-2 OS=Homo sapiens OX=9606 GN=IGHV1-69-2 PE=3 SV=2"
```

To make the data more accessible, we converted the FASTA files to CSV using the [UFPF](https://pypi.org/project/upfp/) package, which splits relevant header information into separate columns.

```python
import glob
import subprocess

# Convert each fasta.gz file to csv
for file in glob.glob('*.fasta.gz'):
	file_root = file.split('.')[0]
	subprocess.run(['upfp-fasta-to-csv', '-g', file, f'{file_root}.csv'])
```

After combining all the individual `*.csv` files in one concatenated one, we need to process and filter selected proteomes files.

```python
# Load the combined CSV file of processed proteomes
df_proteomes_raw = pd.read_csv('combined_unprocessed_csv.txt')

# Add useful annotations
df_proteomes_raw['seq_lenght'] = df_proteomes_raw['sequence'].str.len()
df_proteomes_raw['mnemonic'] = df_proteomes_raw['entry_name'].str.split('_').str[1]

# Generate basic statistics on sequence lengths by species
summaryDFraw=df_proteomes_raw.groupby('mnemonic').agg({'seq_lenght': ['count', 'min', 'median', 'max']})
```

#### Filtering Steps

Some important filtering steps:

1. **Protein Existence Score**: Keep only proteins with a protein existence score of 1 or 2 (PE=1 or PE=2).
2. **Remove Sequences with Rare or Ambiguous Amino Acids**: Exclude any protein sequences containing rare, non-canonical, or ambiguous amino acids. The relevant amino acids are defined by the [DDBJ standards](https://www.ddbj.nig.ac.jp/ddbj/code-e.html):
   - **Ambiguous Amino Acids**:
     - **B**: Aspartic acid or Asparagine
     - **Z**: Glutamic acid or Glutamine
     - **J**: Leucine or Isoleucine
     - **X**: Any amino acid
   - **Rare Amino Acids**:
     - **O**: Pyrrolysine
     - **U**: Selenocysteine
3. **Remove Fragment Entries**: Exclude entries labeled as "Fragment".

```python
df_proteomes_pe_max2 = df_proteomes_raw[df_proteomes_raw['protein_existence'] < 3].reset_index()

# Remove sequences with ambiguous/rare amino acids
df_proteomes_pe_max2_clean = df_proteomes_pe_max2[~df_proteomes_pe_max2['sequence'].str.contains('[BOUXZJ]', regex=True)]

# Remove 'Fragment' entries from the recommended name column
df_proteomes_pe_max2_clean = df_proteomes_pe_max2_clean[~df_proteomes_pe_max2_clean['recommended_name'].str.contains('(Fragment)', regex=False)].copy()

# Save to Feather format
df_proteomes_pe_max2_clean.to_feather('proteomes_4valset_pe_max2_v2b.ftr')

# Generate a column with FASTA formatted strings
df_proteomes_pe_max2_clean['outstring'] = ">" + df_proteomes_pe_max2_clean['accession_number'].astype(str).values + "|" + \
                            df_proteomes_pe_max2_clean['entry_name'].astype(str).values + "\n" + \
                            df_proteomes_pe_max2_clean['sequence'].astype(str).values + "\n"

# Create 'Combined ID' column for merging with other datasets
df_proteomes_pe_max2_clean['combined_id'] = df_proteomes_pe_max2_clean['accession_number'].astype(str).values + "|" + \
                                            df_proteomes_pe_max2_clean['entry_name'].astype(str).values

# Write to FASTA format file
with open('proteomes_4valset_pe2_clean_v2.fa', 'w') as f:
    for index, row in df_proteomes_pe_max2_clean.iterrows():
        f.write(row['outstring'])
```

### SCOP

#### Download SCOP Family Representative Domain Sequences

Obtain the SCOP family representative domain sequences file `scop_fa_represeq_lib_latest.fa` from the [SCOP website](https://www.ebi.ac.uk/pdbe/scop/download). We used version 2022-06-29, downloaded on 2023-11-21.

Convert the FASTA file into a TXT file with the following `fasta2txt.py` script by calling:

```
fasta2txt.py -d scop_fa_represeq_lib_latest_out
```

`fasta2txt.py`:

```python
import argparse
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
# from Bio.Alphabet import IUPAC

parser = argparse.ArgumentParser()
parser.add_argument('filename', help="please provide a fasta file")

parser.add_argument("-d", "--description", help="parse description line instead of ID",
                    action="store_true")

args = parser.parse_args()

file_in = args.filename
file_root = file_in.split('.')[0]

print(file_root)

if args.description:
    descriptionline = 1
else:
    descriptionline = 0


f_out = open(file_root+"_out.txt", "w")

handle = open(file_in, "rU")
for record in SeqIO.parse(handle, "fasta"):
    if descriptionline == 1:
        my_id = record.description #for fasta files with improper ID lines
    else:
        my_id = record.id
    my_seq = record.seq
    my_seq_strip = my_seq.rsplit("*")
    my_name = record.name
    my_description = record.description

    my_new_seq = my_seq_strip[0]


    print(my_id, my_new_seq)

    f_out.write(str(my_id) + '\t' + str(my_new_seq) + '\n')

f_out.close()
```

Run the follwing processing steps.

```python
# Load and process SCOP domain sequences
colnames = ['header', 'sequence']
dfscopfam_fasta_all = pd.read_csv('scop_fa_represeq_lib_latest_out.txt', sep='\t', names=colnames, header=None)

# Function to extract relevant info from SCOP headers
def split_scop_header(x):
    stringlist = x.split()
    id = int(stringlist[0])
    scopfamid = int(stringlist[1].split('=')[1])
    pdbid = stringlist[2].split('=')[1]
    uniid = stringlist[3].split('=')[1]
    return id, scopfamid, pdbid, uniid

# Apply the function to extract SCOP metadata
dfscopfam_fasta_all[['id', 'scopfamid', 'pdbid', 'uniid']] = dfscopfam_fasta_all.apply(lambda x: split_scop_header(x['header']), axis=1, result_type='expand')

# Add sequence length column
dfscopfam_fasta_all['seq_length'] = dfscopfam_fasta_all['sequence'].str.len()

# Create FASTA formatted output column
dfscopfam_fasta_all['outstring'] = ">" + dfscopfam_fasta_all['id'].astype(str).values + "|" + \
                                        dfscopfam_fasta_all['pdbid'].astype(str).values + "|" + \
                                        dfscopfam_fasta_all['uniid'].astype(str).values + "\n" + \
                                        dfscopfam_fasta_all['sequence'].astype(str).values + "\n"

# Filter sequences to remove ambiguous or rare amino acids
dfscopfam_fa_all_clean = dfscopfam_fasta_all[~dfscopfam_fasta_all['sequence'].str.contains('[BOUXZJ]', regex=True)]

# Save the cleaned sequences to a new FASTA file
with open('scop_fa_cleaned_all_for_v2.fa', 'w') as f:
    for index, row in dfscopfam_fa_all_clean.iterrows():
        f.write(row['outstring'])
```

Cluster the SCOP sequences at 30% sequence identity using MMseqs2.

```bash
mmseqs easy-cluster scop_fa_cleaned_all_for_v2.fa scop_fa_v2cluDB30 tmp --min-seq-id 0.3 -s 8 -c 0.8 --cov-mode 3
```

The outputs will include:

- `scop_fa_v2cluDB30_rep_seq.fasta`
- `scop_fa_v2cluDB30_all_seqs.fasta`
- `scop_fa_v2cluDB30_cluster.tsv`

Use the following script to process the cluster representatives.

```python
# Load and process the clustered SCOP representative sequences
colnames = ['header', 'sequence']
dfscop_clu_repseq = pd.read_csv('scop_fa_v2cluDB30_rep_seq_out.txt', header=None, sep='\t', names=colnames)

# Function to extract relevant info from SCOP cluster headers
def split_header(x):
    stringlist = x.split('|')
    id = int(stringlist[0])
    pdbid = stringlist[1]
    uniid = stringlist[2]
    return id, pdbid, uniid

dfscop_clu_repseq[['id', 'pdbid', 'uniid']] = dfscop_clu_repseq.apply(lambda x: split_header(x['header']), axis=1, result_type='expand')

# Create FASTA formatted output column
dfscop_clu_repseq['outstring'] = ">" + dfscop_clu_repseq['header'] + '\n' + \
                                        dfscop_clu_repseq['sequence'] + '\n'
```

Randomly select 10,000 SCOP sequences for the validation set.

```python
# Randomly select 10,000 sequences
dfscop_clu_repseq_10k = dfscop_clu_repseq.sample(n=10000, random_state=1)

# Save the sample to Feather and FASTA formats
dfscop_clu_repseq_10k.to_feather('scop_clu30_valset_10k_v2.ftr')
with open('scop_valset_10k_v2.fa', 'w') as f:
    for index, row in dfscop_clu_repseq_10k.iterrows():
        f.write(row['outstring'])
```

To identify and filter out SCOP sequences that are highly similar to sequences in the reference proteome set, use MMseqs2:

```bash
mmseqs easy-search scop_valset_10k_v2.fa ../proteomes_4valset_pe2_clean_v2.fa scop_10kvalset_vs_refprot_all_v2.m8 tmp -a --cov-mode 3  --format-mode 4 --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,nident"
```

The output file `scop_10kvalset_vs_refprot_all_v2.m8` contains the alignment results, including sequence identities between the two sets. The MMseqs2 argument `"query,...,nident"` adds required output columns in a Blast like format.

Filter out sequences sharing ≥90% sequence identity.

```python
# Load the search results
dfscop_vs_refprot = pd.read_csv('scop_10kvalset_vs_refprot_all_v2.m8', header=0, sep='\t')

# Filter sequences with ≥90% sequence identity
dfscop_vs_refprot_filt = dfscop_vs_refprot[dfscop_vs_refprot['fident'] >= 0.90]

# Extract unique SCOP target IDs
scop_overlap_list = dfscop_vs_refprot_filt['target'].unique().tolist()
```

### OAS

#### Download Paired Antibody Sequence Datasets

To download all the paired antibody sequence datasets from the Observed Antibody Space (OAS), use the script `bulk_download.sh`.

This script is a wrapper for a series of `wget` commands that download individual `csv.gz` files from the [OAS website](https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/).

#### Parse and Process OAS Datasets

The paired antibody dataset is comprised of individual sets, each from a different experiment. Use the script `oas_file_parser.py` to parse and process these files.

- **Filter Out Shorter Sequences**: The script removes antibody sequences where both heavy and light chains have the ANARCI\_ status `Shorter`. Sequences that are not shorter in both chains (A and B) are retained.
- **Column Selection**: The script keeps only the relevant columns for each antibody sequence: sequence ID, heavy chain (HC), light chain (LC) amino acid sequences, and the sequences for all LC and HC CDR (complementarity-determining region) sequences.

- **Unique ID Generation**: A unique ID (UUID) is generated for each paired antibody to ensure uniformity across different datasets.

- **Concatenation of Sequences**:

  - The light and heavy chain amino acid sequences are concatenated in two formats:
    - LC|HC (l2h)
    - HC|LC (h2l)
  - The chain break marker `|` is used between LC and HC.
  - The concatenated sequences are labeled with the UUID and suffixes "\_A" for HC|LC and "\_B" for LC|HC.

- **Concatenation for MMseqs2 Clustering**: The heavy and light chain sequences are concatenated in the format `HC + XXX + LC` where `X` is an unknown amino acid placeholder to separate the chains.

- **Output Files**:

  - A compressed `CSV` file is created for all processed datasets: `_oas_paired_combined_processed_csv.txt.gz_`.
  - Sequences are also written out in `fasta` format.

  _(Note: The script also has the option to concatenate individual CDR amino acid sequences of each antibody, but this is not used in the current setup.)_

`oas_file_parser.py` :

```python
#!/usr/bin/env python
# Lacks check for ambigious and rare amino acids [BOUXZJ]. Verify afterwards
import pandas as pd
import numpy as np
import re
import glob
import subprocess
import uuid

columns_to_keep2 = ['sequence_id_heavy',
		'sequence_alignment_aa_heavy',
		'sequence_id_light',
		'sequence_alignment_aa_light',
		'cdr1_aa_heavy',
		'cdr2_aa_heavy',
		'cdr3_aa_heavy',
		'cdr1_aa_light',
		'cdr2_aa_light',
		'cdr3_aa_light',
		]

# Function  to concatenate heavy and light chain sequences in both h2l and l2h orientations, and CDR loop sequences only
def sequence_concats(row):
	full_h2l = ">" + df['uuid'].astype(str).values + "|" + "A" + "\n" + \
								df['sequence_alignment_aa_heavy'].astype(str).values + "|" + \
								df['sequence_alignment_aa_light'].astype(str).values + "\n"

	full_l2h = ">" + df['uuid'].astype(str).values + "|" + "B" + "\n" + \
								df['sequence_alignment_aa_light'].astype(str).values + "|" + \
								df['sequence_alignment_aa_heavy'].astype(str).values + "\n"

	cdr_concat = ">" + df['uuid'].astype(str).values + "|" + "CDRs" + "\n" + \
								df['cdr1_aa_heavy'].astype(str).values + "X" + \
								df['cdr2_aa_heavy'].astype(str).values + "X" + \
								df['cdr3_aa_heavy'].astype(str).values + "XX" + \
								df['cdr1_aa_light'].astype(str).values + "X" + \
								df['cdr2_aa_light'].astype(str).values + "X" + \
								df['cdr3_aa_light'].astype(str).values + "X" + "\n"
	return full_h2l, full_l2h, cdr_concat

def fasta_writer(fname, df, outstring):
	with open(fname, 'w') as f:
		for index, row in df.iterrows():
			# print(row['outstring'])
			f.write(row[outstring])

for file in glob.glob('*.csv.gz'):
	file_root = file.split('.')[0]
	dfin = pd.read_csv(file, header=1, compression='gzip')
	##not shorter A *and* not shorter B
	dfclean = dfin.query("~ANARCI_status_heavy.str.contains('Shorter') \
							 and ~ANARCI_status_light.str.contains('Shorter')")
    ##To do: add check for [BOUXZJ]
	dffiltered = dfclean[columns_to_keep2]
	df = dffiltered.copy(deep=True)
	df['file_source'] = str(file_root)
	df['uuid'] = [uuid.uuid4() for _ in range(len(df.index))]
# 	df[['full_h2l', 'full_l2h', 'cdr_concat']] =  df.apply(sequence_concats, axis=1)
	df['full_h2l'] = ">" + df['uuid'].astype(str).values + "|" + "_A" + "\n" + \
							df['sequence_alignment_aa_heavy'].astype(str).values + "|" + \
							df['sequence_alignment_aa_light'].astype(str).values + "\n"

	df['full_l2h'] = ">" + df['uuid'].astype(str).values + "|" + "_B" + "\n" + \
								df['sequence_alignment_aa_light'].astype(str).values + "|" + \
								df['sequence_alignment_aa_heavy'].astype(str).values + "\n"

	df['cdr_concat'] = ">" + df['uuid'].astype(str).values + "|" + "_CDRs" + "\n" + \
								df['cdr1_aa_heavy'].astype(str).values + "X" + \
								df['cdr2_aa_heavy'].astype(str).values + "X" + \
								df['cdr3_aa_heavy'].astype(str).values + "XX" + \
								df['cdr1_aa_light'].astype(str).values + "X" + \
								df['cdr2_aa_light'].astype(str).values + "X" + \
								df['cdr3_aa_light'].astype(str).values + "X" + "\n"

	df['mmseq'] = ">" + df['uuid'].astype(str).values + "|" + "_A" + "\n" + \
							df['sequence_alignment_aa_heavy'].astype(str).values + "XXX" + \
							df['sequence_alignment_aa_light'].astype(str).values + "\n"

	fout = file_root + "_processed.csv"
	df.to_csv(fout)
	fasta_writer(f'{file_root}_full_h2l.fasta', df, 'full_h2l')
	fasta_writer(f'{file_root}_full_l2h.fasta', df, 'full_l2h')
	fasta_writer(f'{file_root}_cdr_concat.fasta', df, 'cdr_concat')
	fasta_writer(f'{file_root}_mmseq.fasta', df, 'mmseq')
```

#### Cluster OAS Sequences Using MMseqs2

use MMseqs2 to cluster all the sequences:

To cluster the sequences, use MMseqs2 with the following command:

```bash
mmseqs easy-linclust oas_paired_4mmseqs_combined.fasta oas_paired_linclustDB90 tmp --min-seq-id 0.9
```

#### Create a 10K Sample of Representative Sequences for Validation

After clustering, convert the resulting FASTA file to TXT format with the `fasta2txt.py` script. Create a sample of 10K sequences from the clustered dataset.

```python
# Import clustered sequences into a pandas dataframe
colnames = ['id', 'sequence']
dfclustered = pd.read_csv('oas_paired_linclustDB90_rep_seq_out.txt', sep='\t', names=colnames)

headernames = ['sequence_id_heavy','sequence_alignment_aa_heavy','sequence_id_light',
               'sequence_alignment_aa_light','cdr1_aa_heavy','cdr2_aa_heavy','cdr3_aa_heavy',
               'cdr1_aa_light','cdr2_aa_light','cdr3_aa_light','file_source','uuid','full_h2l',
               'full_l2h','cdr_concat','mmseq']

dfoas_all = pd.read_csv('oas_paired_combined_processed_csv.txt.gz', names=headernames, compression='gzip')

# Extract UUID by splitting the sequence ID
dfclustered['uuid'] = dfclustered['id'].str.split(pat='|',expand=True)[0]

# Sample 10K sequences
clustered_10ksample_id_list = dfclustered.sample(n=10000, random_state=1)['uuid'].to_list()


# Create a dataframe of the 10K sampled sequence with all property colums
df_clustered_full_10k = dfoas_all[dfoas_all['uuid'].isin(clustered_10ksample_id_list)]

# Verify that 10 K sampled sequences do not contain rare or ambigious amino acids (step lacks in oas_file_parser.py)
df_clustered_full_10k[~df_clustered_full_10k['sequence_alignment_aa_light'].str.contains('[BOUXZJ]', regex=True)]
df_clustered_full_10k[~df_clustered_full_10k['sequence_alignment_aa_heavy'].str.contains('[BOUXZJ]', regex=True)]


# Save the sampled sequences to a CSV file
df_clustered_full_10k.to_csv('oas_paired_clustered_10ksample_v2.csv')

# Write out sequences in fasta format
with open('oas_paired_10k_bothdirections_v2.fa', 'w') as f:
    for index, row in df_clustered_full_10k.iterrows():
        f.write(row['full_h2l'])
        f.write(row['full_l2h'])


# Create a set of OAS sequences that excludes the sequences in the 10k validation set (used for Train set construction)
dfoas_all_novalset = dfoas_all[~dfoas_all['uuid'].isin(clustered_10ksample_id_list)]

# Verify that sequences do not contain rare or ambigious amino acids (step lacks in oas_file_parser.py)
dfoas_all_novalset[dfoas_all_novalset['sequence_alignment_aa_heavy'].str.contains('[BOUXZJ]', regex=True)]
dfoas_all_novalset[dfoas_all_novalset['sequence_alignment_aa_light'].str.contains('[BOUXZJ]', regex=True)]

```

### Filter Validation Sets

#### Remove SCOP Overlap with Cleaned Reference Proteome Set

Use MMseqs2 to search the SCOP validation set against the cleaned reference proteome set to find similar sequences.

```bash
mmseqs easy-search scop_valset_10k_v2.fa ../proteomes_4valset_pe2_clean_v2.fa scop_10kvalset_vs_refprot_all_v2.m8 tmp -a --cov-mode 3 --format-mode 4 --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,nident"
```

The output `scop_10kvalset_vs_refprot_all_v2.m8` contains the overlap between SCOP IDs and reference proteome IDs.

Load the MMSEQS search results and filter sequences that have a sequence identity of 90% or higher.

```python
# Load SCOP overlap file and filter sequences with >= 90% identity
scopoverlapfile = 'scop_10kvalset_vs_refprot_all_v2.m8'
dfscop_vs_refprot = pd.read_csv(scopoverlapfile, header=0, sep='\t')
dfscop_vs_refprot_filt = dfscop_vs_refprot[dfscop_vs_refprot['fident'] >= 0.90]

# Create a list of overlapping reference proteome IDs
scop_overlap_list = dfscop_vs_refprot_filt['target'].unique().tolist()
```

#### Remove OAS Overlap with Cleaned Reference Proteome Set

To deduplicate the OAS sequences against the reference proteome, create a search file that separates the light and heavy chains:

```python
# Write the validation sequences in single direction format for MMseqs2 search
with open('oas_paired_10k_single_direction_4search_v2.fa', 'w') as f:
    for index, row in df_clustered_full_10k.iterrows():
        f.write(">")
        f.write(row['uuid'] + "_lc" + "\n")
        f.write(row['sequence_alignment_aa_light'] + "\n")
        f.write(">")
        f.write(row['uuid'] + "_hc" + "\n")
        f.write(row['sequence_alignment_aa_heavy'] + "\n")
```

Next, use MMseqs2 to search for highly similar sequences in the cleaned reference proteome set using the OAS validation set.

```bash
mmseqs easy-search oas_paired_10k_single_direction_4search_v2.fa ../proteomes_4valset_pe2_clean_v2.fa oas_rep_vs_refprot_all_v2.m8 tmp -a --cov-mode 3  --format-mode 4 --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,nident"
```

The output `oas_rep_vs_refprot_all_v2.m8` contains the overlap between OAS IDs and reference proteome IDs.

Load the OAS overlap results and filter sequences with a sequence identity of 90% or higher.

```python
# Load OAS overlap file and filter sequences with >= 90% identity
oasoverlapfile = 'oas_rep_vs_refprot_all_v2.m8'
dfoas_vs_refprot = pd.read_csv(oasoverlapfile, sep='\t')
dfoas_vs_refprot_filt = dfoas_vs_refprot[dfoas_vs_refprot['fident'] >= 0.90]

# Create a list of overlapping reference proteome IDs
oas_overlap_list = dfoas_vs_refprot_filt['target'].unique().tolist()
```

#### Combine OAS and SCOP Overlap Lists

Now, combine the overlap lists from OAS and SCOP and remove the overlapping sequences from the cleaned reference proteome.

```python
# Combine OAS and SCOP overlap lists
total_overlap = oas_overlap_list + scop_overlap_list
total_overlap_unique = list(set(total_overlap))

# Remove overlapping sequences from the cleaned reference proteome
dfref_prot_filtered = df_proteomes_pe_max2_clean[~df_proteomes_pe_max2_clean['combined_id'].isin(total_overlap_unique)]
```

#### Filter Out Short Sequences

Remove sequences that are 5 or fewer amino acids in length, as they are likely to be artifactual proteins.

```python
# Remove sequences with 5 or fewer amino acids
dfref_prot_filtered = dfref_prot_filtered[dfref_prot_filtered['seq_lenght'] > 5].copy()
```

#### Sample 10K Sequences and Save as Reference Proteome Validation Set

Create a random sample of 10K sequences from the filtered reference proteome and save the sample in both `feather` and `fasta` formats.

```python
# Sample 10K sequences
dfref_prot_filtered_10k_sample = dfref_prot_filtered.sample(n=10000, random_state=1)

# Save as feather file
refprotfilter_milavalidation_10k_v2 = 'refproteome_filtered_validationset_10k_v2.ftr'
dfref_prot_filtered_10k_sample.to_feather(refprotfilter_milavalidation_10k_v2)

# Export as FASTA file
refprot_filter_validation_10k_v2_fa = refprot_cluster_path / 'refproteome_filtered_validationset_10k_v2.fasta'
with open(refprot_filter_validation_10k_v2_fa, 'w') as f:
    for index, row in dfref_prot_filtered_10k_sample.iterrows():
        f.write(row['outstring'])
```

## Train Sets

### UniRef

#### Download UniRef100

To download the current version of UniRef100, you can directly download the `fasta.gz` file using the following command:

```bash
wget -c https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz
```

For older UniRef100 releases, file formats may differ by year. Recent releases are stored in XML format, which must be converted into FASTA format using UniRef's provided script, `unirefxml2fasta.pl`. Some releases also offer a metalink file for downloading. To accelerate downloads, consider using `aria2c`:

```bash
aria2c -c [metalink file]
```

#### Filter UniRef100

To remove sequences with ambiguous or non-frequent amino acids (X, B, O, U, Z, and J), use `seqkit`:

```bash
# Remove ambiguous sequences
seqkit grep -s uniref100.fasta.gz -i -r  -p [XBOUZJ] -v -o uniref100_no_ambig_aa.fasta.gz

# Count original sequences
zcat uniref100.fasta.gz | grep ">" | wc -l > uniref100_size.txt

# Count non-ambiguous sequences
zcat uniref100_no_ambig_aa.fasta.gz | grep ">" | wc -l > uniref100_no_ambig_aa_size.txt
```

Search non-ambiguous sequences against a combined fasta of all three validation sets (UniProt reference proteomes, SCOP, and OAS) using MMseqs2.

```bash
mmseqs easy-search combined_validation_sets_4dedup_trainingset.fasta ../uniref/uniref100_no_ambig_aa.fasta.gz valset_uniref100search_mode_a_covmode0_v2.m8 tmp -a --cov-mode 0 --format-mode 4 --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,nident"

# Compress results
gzip valset_uniref100search_mode_a_covmode0_v2.m8
```

Using python, tab-separated MMSEQS2 results file and extract unique IDs with sequence id that have a sequence identity >90% to any validation set sequence. We drop duplicates here because for IDs that hit multiple validation set targets it doesn't matter which one is hit, we just keep one row as confirmation that a hit occurred. The output is a text file that contains the UniRef100 IDs to remove.

```python
import pandas as pd

# Load the MMSEQS2 results file
dedubuniref100='valset_uniref100search_mode_a_covmode0_v2.m8.gz'
dfuni100_dups = pd.read_csv(dedubuniref100, compression='gzip', sep='\t')

# Filter sequences with >= 90% identity
dfuni100dedup90 = dfuni100_dups[dfuni100_dups['fident'] >= 0.9]

# Remove duplicate target IDs
dfuni100dedup90_unique = dfuni100dedup90.drop_duplicates(subset=['target'])

# Write the filtered UniRef100 IDs to a text file
with open('uniref100-id-valset-at90-v2.txt', 'w') as f:
    f.write(dfuni100dedup90_unique['target'].str.cat(sep='\n'))
    f.write("\n") # Ensure no '%' sign appears in the last line
```

Use `seqkit` to generate a filtered UniRef100 FASTA file by removing sequences that are listed in the exclusion list (`uniref100-id-valset-at90-v2.txt`), which contains IDs with high similarity to the validation sets. Note that `zcat` commands simply collect dataset sizes before and after processing.

```bash
# Remove sequences listed in the exclusion file
seqkit grep -f uniref100-id-valset-at90-v2.txt ../uniref/uniref100_no_ambig_aa.fasta.gz -v -o uniref100_no_ambig_aa_dedupval90_v2.fa.gz

# Count sequences before and after filtering
zcat ../uniref/uniref100_no_ambig_aa.fasta.gz | grep ">" | wc -l > uniref100_noambigaa_size.txt
zcat uniref100_no_ambig_aa_dedupval90_v2.fa.gz | grep ">" | wc -l > uniref100_noambig_aa_dedupval90_v2_size.txt
```

### OAS

#### Filter OAS

For OAS sequences, we perform deduplication at 99% sequence identity. This is necessary due to the high similarity across germline frameworks and we are primarily interested in minor variations in CDR loop sequences.

Use MMseqs2 to search the OAS paired sequence dataset against the OAS validation set:


```bash
mmseqs easy-search oas_paired_10k_single_direction_4search_v2.fa oas_paired_all_valremoved_single_direction_4search_v2.fa oas_valset_vs_oas_paired_all_valremoved_singledirection_mode_a_covmode0_v2_s4.m8 tmp2 -a --cov-mode 0 --format-mode 4 --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,nident"
```

Use Python to read the MMseqs2 output and manually filter sequences. Sequences with >99% sequence identity are excluded from the training set.

```python
oas_overlap = 'oas_valset_vs_oas_paired_all_valremoved_singledirection_mode_a_covmode0_v2_s4.m8.gz'

dfoas_all_novalset_vs_valset = pd.read_csv(oas_overlap, sep='\t', header=0, compression='gzip')

# Function to split uuid part from the chain identifier and return uuid
def split_id(x):
    stringlist = x.split('_')
    base_id = stringlist[0]
    return base_id

dfoas_all_novalset_vs_valset['target_baseid']= dfoas_all_novalset_vs_valset.apply(lambda x: split_id(x['target']), axis=1)

# Filter sequences with >99% identity
dfoas_all_novalset_vs_valset_dedup99 = dfoas_all_novalset_vs_valset[dfoas_all_novalset_vs_valset['fident'] > 0.99]
dfoas_all_novalset_vs_valset_dedup99_unique = dfoas_all_novalset_vs_valset_dedup99.drop_duplicates(subset=['target_baseid'], keep='first')

# Create a list of unique IDs to remove
oas_dedup99list_unique = dfoas_all_novalset_vs_valset_dedup99_unique['target_baseid'].to_list()

# Exclude sequences from the training set
dfoas_all_dedup99 = dfoas_all_novalset[~dfoas_all_novalset['uuid'].isin(oas_dedup99list_unique)]

# Save the deduplicated dataset as a Feather file
dfoas_all_dedup99.to_feather('oas_all_paired_dedubbed_at99_train_v2.ftr')

# Export the deduplicated dataset in FASTA format; in both h2l and l2h orientation
with open('oas_paired_all_train_bothdirections_dedupped_at99seqid_v2.fa', 'w') as f:
    for index, row in dfoas_all_dedup99.iterrows():
        f.write(row['full_h2l'])
        f.write(row['full_l2h'])
```

### SCOP

#### Filter SCOP

For SCOP sequences, we perform deduplication by searching the SCOP validation set against the full SCOP representative sequence library and filtering sequences with >90% SeqID.

```bash
mmseqs easy-search scop_valset_mila_10k_v2.fa scop_fa_represeq_lib_latest.fa scop_valset_vs_scop_repseq__mode_a_covmode0_v2.m8 tmp -a --cov-mode 0 --format-mode 4 --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,nident"
```

Use Python to read the MMseqs2 output and filter sequences with >90% sequence identity. Create a list of SCOP IDs to exclude from the training set.

```python
# Load MMseqs2 results
scop_val_vs_all = 'scop_valset_vs_scop_repseq__mode_a_covmode0_v2.m8.gz'
dfscop_dups = pd.read_csv(scop_val_vs_all, compression='gzip', sep='\t')

# Filter sequences with >= 90% identity
dfscopdedup90 = dfscop_dups[dfscop_dups['fident'] >= 0.9]

# Remove duplicate target IDs
dfscopdedup90_unique = dfscopdedup90.drop_duplicates(subset=['target'])

# Save filtered SCOP IDs to a text file
with open('scop-id-valset-at90-v2.txt', 'w') as f:
    f.write(dfscopdedup90_unique['target'].astype(str).str.cat(sep='\n'))
    f.write("\n")
```

Use `seqkit` to remove the SCOP IDs identified in the exclusion list (`scop-id-valset-at90-v2.txt`) from the SCOP representative sequence library and create a deduplicated SCOP training set.

```bash
# Remove duplicates from SCOP representative set
seqkit grep -f scop-id-valset-at90-v2.txt scop_fa_represeq_lib_latest.fa -v -o scop_fa_repseq_train_dedupval90_v2.fa.gz

# Count sequences before and after filtering
cat scop_fa_represeq_lib_latest.fa | grep ">" | wc -l > scop_fa_repseq_size.txt
zcat scop_fa_repseq_train_dedupval90_v2.fa.gz | grep ">" | wc -l > scop_fa_repseq_dedupval90_v2_size.txt
```
