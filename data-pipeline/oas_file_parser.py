#!/usr/bin/env python
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
	
