def remap():
    return {'female': 'Female sex assigned at birth, n (%)',
            'White/Caucasian': 'White race, n (%)',
            'Hispanic': 'Hispanic ethnicity, n (%)',
            'Black/African American': 'Black race, n (%)',
            'Unknown': 'Unknown, n (%)',
            'Two or more races': 'Mixed race, n (%)',
            'Asian': 'Asian race, n (%)',
            'Age': 'Age, years, mean (SD)',
            'number_of_comorbidites': 'Comorbidities, mean (SD)'
            }

def count_perc_diagnosis(df, overall_count,
                        diagnosis1='No Diagnosis Given', diagnosis2='Reading Impairment'):
    # get filtered numbers
    no_diagnosis_count = sum(df['Diagnosis']==diagnosis1)
    diagnosis_count = sum(df['Diagnosis']==diagnosis2)
    all_count = no_diagnosis_count + diagnosis_count

    # percent of overall
    no_diagnosis_perc = no_diagnosis_count / (all_count)*100
    diagnosis_perc = diagnosis_count / (all_count)*100
    all_perc = all_count / (overall_count)*100

    data_dict = {
                'Overall': f'{round(all_count, 1)} ({round(all_perc, 1)})',
                f'{diagnosis1}': f'{round(no_diagnosis_count, 1)} ({round(no_diagnosis_perc, 1)})',
                f'{diagnosis2}': f'{round(diagnosis_count, 1)} ({round(diagnosis_perc, 1)})'
                 }
    
    return data_dict


def overall_numbers_diagnosis(df, diagnosis1='No Diagnosis Given', diagnosis2='Reading Impairment'):
    # get overall numbers
    overall_count = df.shape[0]
    overall_no_diagnosis = sum(df['Diagnosis']==diagnosis1)
    overall_diagnosis = sum(df['Diagnosis']==diagnosis2)

    return overall_count, overall_no_diagnosis, overall_diagnosis


def make_table_count(df,
                    cols,
                    vals,
                    diagnosis1='No Diagnosis Given', 
                    diagnosis2='Reading Impairment'
                    ):
    from collections import defaultdict
    import pandas as pd

    # get patient count
    df_count = df.groupby('Identifiers').head(1)

    # get overall count
    overall_count, overall_no_diagnosis, overall_diagnosis = overall_numbers_diagnosis(df, diagnosis1, diagnosis2)

    # loop over races and ethnicities
    out_dict = defaultdict(list)
    for (col, val) in zip(cols, vals): 

        # filter dataframe
        df_filter = df_count[df_count[col]==val].reset_index(drop=True)

        # get patient count and % of overall
        data_dict = count_perc_diagnosis(df=df_filter,
                                         overall_count=overall_count,
                                         diagnosis1=diagnosis1,
                                         diagnosis2=diagnosis2
                                         )
        
        for k,v in data_dict.items():
            out_dict[k].append(v)
        
    # make dataframe and clean up
    df_out = pd.DataFrame.from_dict(out_dict)
    df_out['Demographics'] = vals
    df_out['Demographics'] = df_out['Demographics'].map(remap())
    df_out.loc[len(df_out)] = [overall_count, overall_no_diagnosis, overall_diagnosis, 'n']

    return df_out
    

def make_table_mean_std(df, 
                        cols=['Age'], 
                        cols_new=['Age, years (mean, SD)'],
                        diagnosis1='No Diagnosis Given',
                        diagnosis2='Reading Impairment'
                        ):
    import pandas as pd
    from collections import defaultdict
    
    # add mean, std
    out_dict = defaultdict(list)
    for col, col_new in zip(cols, cols_new):

        mean_overall = df[col].mean().round(2)
        std_overall = df[col].std().round(2)

        no_diagnosis = df[df['Diagnosis']==diagnosis1][col]
        diagnosis = df[df['Diagnosis']==diagnosis2][col]

        data_dict = {'Overall': f'{mean_overall} ({std_overall})',
                    f'{diagnosis1}': f'{no_diagnosis.mean().round(2)} ({no_diagnosis.std().round(2)})',
                    f'{diagnosis2}': f'{diagnosis.mean().round(2)} ({diagnosis.std().round(2)})',
                    'Demographics': col_new
                    }
        
        for k, v in data_dict.items():
            out_dict[k].append(v)

    df_out = pd.DataFrame.from_dict(out_dict)

    return df_out

def make_table_png(df, outpath):
    import matplotlib.pyplot as plt
    import pandas as pd
    from pandas.plotting import table

    fig, ax = plt.subplots()
    ax.axis('off')

    table = pd.plotting.table(ax, df, loc='center', cellLoc='center')

    plt.savefig(outpath)
    