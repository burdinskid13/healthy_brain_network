import click
import warnings
warnings.filterwarnings("ignore")

def run(
    assessments=['Child', 'Parent'], 
    data_type='raw', 
    transformer='distilbert-base-nli-mean-tokens'
    ):
    import os
    import pandas as pd
    import numpy as np
    from hbn.constants import Defaults
    from hbn.models import item_analysis
    from hbn.scripts import make_files
    from hbn import io

    # make data files
    #make_files.make_data_files()

    # load data
    df_data, df_dict, df_diagnosis = item_analysis.load_data(
                                            assessments=assessments, 
                                            data_type=data_type
                                            )


    # calculate similarity
    sentences = df_dict['questions']
    cosine_scores, pairs = item_analysis.sentence_similarity(sentences, transformer=transformer)

    # cosine scores
    df1 = pd.DataFrame(np.array(cosine_scores))

    # score pairs
    df2 = pd.DataFrame()
    for idx, pair in enumerate(pairs):
        df2.loc[idx, 'idx1'] = pairs[idx]['index'][0]
        df2.loc[idx, 'idx2'] = pairs[idx]['index'][1]
        df2.loc[idx, 'score'] = pairs[idx]['score'].tolist()

    data_dict = {'questions': df_dict,
                'diagnosis': df_diagnosis,
                'cosine_scores': df1,
                'pairs': df2,
                'transformer': transformer
                }

    # save as hdf5
    fname = 'sentence-similarity' + '_' + '_'.join(assessments) + '_' + data_type + '.h5'
    io.save_dict_as_hdf5(fpath=os.path.join(Defaults.SUBTYPE_DIR, fname), data_dict=data_dict)

if __name__ == "__main__":
    run()
