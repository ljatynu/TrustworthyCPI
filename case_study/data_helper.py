import numpy as np
import pandas as pd

from MyUtils.data_helpers import txt_Str_data_to_IndexVec_npy

if __name__ == '__main__':
    # Get the encoding vector of C-P pair from text sequence
    input_path = 'antiviral_data/existing_drugs_3CLPro_pair.txt'
    data = pd.read_csv(input_path, header=None)
    IndexVec_data=txt_Str_data_to_IndexVec_npy(input_path)
    np.save('existing_drugs_3CLPro_pair_IndexVec_data.npy',IndexVec_data)
    pass


