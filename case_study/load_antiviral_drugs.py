import os

import pandas as pd
import requests


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def load_antiviral_drugs(path = 'antiviral_data', no_cid = False):
	url = 'https://dataverse.harvard.edu/api/access/datafile/4159652'
	if not os.path.exists(path):
	    os.makedirs(path)
	download_path = os.path.join(path, 'antiviral_drugs.tab')
	download_url(url, download_path)
	df = pd.read_csv(download_path, sep = '\t')
	if no_cid:
		return df.SMILES.values, df[' Name'].values
	else:
		return df.SMILES.values, df[' Name'].values, df['Pubchem CID'].values
if __name__ == '__main__':
	load_antiviral_drugs()
