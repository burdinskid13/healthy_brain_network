import os
import re
import glob
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

from hbn.constants import Defaults


def make_item_names():
    # read file
    with open(os.path.join(Defaults.PHENO_DIR, 'item-names.csv'), "r") as f:
        data = [re.sub(r"�+", "'", l).strip().split(";", -1) for l in f.readlines()]

    # make pandas dataframe
    questions = []
    keys = []
    for d in data:

        questions.append(' '.join(d[:-1]))
        if len(d)>1:
            keys.append(d[-1])
        else:
            keys.append(None)

    df = pd.DataFrame(np.array([questions, keys]).T, columns=['questions', 'keys'])

    # get all data dictionaries
    data_dir = os.path.join(Defaults.PHENO_DIR, 'Release9_DataDic')
    os.chdir(data_dir)
    data_dics = glob.glob('*xlsx')

    dicts = {}
    for data_dic in data_dics:
        if '~$' not in data_dic:
            df_dict = pd.read_excel(data_dic)
            dict_list = df_dict.to_numpy().flatten()
            key = data_dic.strip('.xlsx')
            dicts.update({key: dict_list})

    keys_mat = np.array(np.zeros((len(df),2)), dtype=object)
    for idx in df.index:
        datadics = []
        # loop over dictionary keys
        for k,v in dicts.items():
            if df.loc[idx, 'keys'] in v:
                datadics.append(k)
        if 0<len(datadics)<=2:
            keys_mat[idx,:] = datadics
        else:
            keys_mat[idx,:] = 'No Key'

    # now loop over `keys` and link keys to data dictionary
    for idx in df.index:

        similarity = SequenceMatcher(None, keys_mat[idx,0], keys_mat[idx,1]).ratio()

        # if keys have two corresponding measures
        # check which sentences match and return most likely measure
        if similarity==1:
            df.loc[idx,'datadic'] = keys_mat[idx][0]
        else:
            removelist = " "
            question = re.sub(r'[^\w'+removelist+']', '', df.loc[idx, 'questions'])
            ratios = {}
            for kk in keys_mat[idx]:
                for vv in dicts[kk]:
                    print(f'comparing {question} to {vv}')
                    try:
                        # strip non-alphanumeric characters from strings and compare
                        compare_str = re.sub(r'[^\w'+removelist+']', '', vv)
                        ratio = SequenceMatcher(None, question, compare_str).ratio()
                        ratios.update({f'{kk}:{compare_str}': ratio})
                    except:
                        pass
            # find matching sentence
            sentence_idx = np.argmax(list(ratios.values()))
            correct_key, correct_sentence = list(ratios.keys())[sentence_idx].split(':')
            #df.loc[idx, 'new_question'] = correct_sentence
            df.loc[idx, 'datadic'] = correct_key

    # make new column names
    df = _match_datadic_to_data(dataframe=df)

    # save out new file
    df.to_csv(os.path.join(Defaults.PHENO_DIR, 'item-names-new.csv'), index=False)


def make_data_files():
    import os
    from hbn.features import feature_selection
    from hbn.constants import Defaults
    from hbn import io

    ## save out data files (link with dictionary keys) for different feature specs
    assessments = ['Parent', 'Child', 'Teacher']
    data_dict = {True: 'preprocessed', False: 'raw'}

    for assessment in assessments:
        for k,v in data_dict.items():
        
            feature_spec = os.path.join(Defaults.FEATURE_DIR, f'features-{assessment}_Measures-all-all-all-spec.json')

            # preprocessed and raw data
            feature_info = io.read_json(feature_spec)
            feature_info['preprocessing']['preprocess'] = k
            df = feature_selection.phenotype_features(feature_spec=feature_info, drop_identifiers=False)
            df.columns = [col.replace("numeric__", "") for col in df.columns]
            
            df.to_csv(os.path.join(Defaults.SUBTYPE_DIR, f'{assessment}-features-{v}.csv'), index=False)
            print(f'saving out {assessment}-features-{v}.csv to disk')


def _match_datadic_to_data(dataframe):
    from hbn import io
    import glob
    from hbn.constants import Defaults
    from collections import defaultdict

    # grab all feature files and make dictionary from abbrevs and datadic args
    feature_specs = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*features*'))

    # initializing dict with lists
    new_dict = defaultdict(list)

    # loop over feature specs
    for spec in feature_specs:
        if 'features-parent_spec' not in spec:
            info = io.read_json(spec)

            dict = {info['datadic']: [info['abbrevs'], info['assessment'], info['domains'], info['measures']]}
        for k,v in dict.items():
            new_dict[k].append(v)

    # assign column names so that data can be indexed correctly
    for index in dataframe.index:
        key = dataframe.loc[index, 'datadic']
        if key in new_dict:
            dataframe.loc[index, 'col_name'] = new_dict[key][0][0] + ',' + dataframe.loc[index, 'keys']
            dataframe.loc[index, 'assessment'] = new_dict[key][0][1]
            dataframe.loc[index, 'domains'] = new_dict[key][0][2]
            dataframe.loc[index, 'measures'] = new_dict[key][0][3]

    return dataframe


def run():
    
    # make data files
    make_data_files()

    # make item names (data dict)
    make_item_names()
    
if __name__ == "__main__":
    run()