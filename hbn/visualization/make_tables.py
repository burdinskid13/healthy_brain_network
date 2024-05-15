def remap(vals):
    map = {'female': 'Female sex assigned at birth, n (%)',
            'White/Caucasian': 'White race, n (%)',
            'Hispanic': 'Hispanic ethnicity, n (%)',
            'Black/African American': 'Black race, n (%)',
            'Unknown': 'Unknown, n (%)',
            'Two or more races': 'Mixed race, n (%)',
            'Asian': 'Asian race, n (%)',
            'Age': 'Age, years, mean (SD)',
            'number_of_comorbidites': 'Comorbidities, mean (SD)'
            }
    # programmatically reformat diagnosis
    data_dict = {}
    for val in vals:
        if val in map.keys():
            data_dict.update({val: map[val]})
        else:
            data_dict.update({val: f'{val}, n (%)'})

    return data_dict

def count_perc_vals(df, overall_count, col='Diagnosis', vals=['No Diagnosis Given', 'Reading Impairment']):

    # get counts
    counts = []
    for val in vals:
        count = sum(df[col]==val)
        counts.append(count)
    all_count = sum(counts)

    # get percentages
    percs = []
    for v in counts:
        percs.append(v / (all_count)*100)
    all_perc = all_count/ (overall_count)*100

    out_dict = {'Overall': f'{round(all_count, 1)} ({round(all_perc, 1)})'}
    for name, count, perc in zip(vals, counts, percs):
        out_dict.update({f'{name}': f'{round(count, 1)} ({round(perc, 1)})'})

    return out_dict


def overall_numbers(df, col, vals=['No Diagnosis Given', 'Reading Impairment']):
    # get overall numbers for `overall`
    overall_count = df.shape[0]
    # loop over vals
    counts = []
    for val in vals:
        counts.append(sum(df[col]==val))

    out_list = [overall_count] + counts

    return out_list


def make_table_count(df,
                    cols,
                    vals,
                    split='Diagnosis',
                    split_vals=['No Diagnosis Given', 'Reading Impairment'],
                    overall={'DX_Cat_Name': 'Reading Impairment'}
                    ):
    from collections import defaultdict
    import pandas as pd

    if 'Identifiers' in df.columns:
        # get patient count
        df_count = df.groupby('Identifiers').head(1)
    else:
        df_count = df

    # get overall count
    if overall is not None:
        col = list(overall.keys())[0]
        val = list(overall.values())[0]
        overall_counts = overall_numbers(df=df_count[df_count[col]==val], col=split, vals=split_vals)
    else:
        overall_counts = overall_numbers(df=df_count, col=split, vals=split_vals)

    # loop over cols + corresponding vals
    out_dict = defaultdict(list)
    for (col, val) in zip(cols, vals): 

        # filter dataframe
        df_filter = df_count[df_count[col]==val].reset_index(drop=True)

        # get patient count and % of overall
        data_dict = count_perc_vals(df=df_filter,
                                    overall_count=overall_counts[0], # first item in overall counts should be `overall_count``
                                    col=split, 
                                    vals=split_vals
                                    )
        
        for k,v in data_dict.items():
            out_dict[k].append(v)
        
    # make dataframe and clean up
    df_out = pd.DataFrame.from_dict(out_dict)
    df_out['Demographics'] = vals
    df_out['Demographics'] = df_out['Demographics'].map(remap(vals))
    first_row = pd.DataFrame([pd.Series(overall_counts + ['n'], index=df_out.columns)])
    df_out = pd.concat([first_row, df_out], ignore_index=True)

    return df_out
    

def make_table_mean_std(df, 
                        cols=['Age'], 
                        cols_new=['Age, years (mean, SD)'],
                        split='Diagnosis',
                        split_vals=['No Diagnosis Given', 'Reading Impairment']
                        ):
    import pandas as pd
    from collections import defaultdict
    
    # add mean, std
    out_dict = defaultdict(list)
    for col, col_new in zip(cols, cols_new):

        mean_overall = df[col].mean().round(2)
        std_overall = df[col].std().round(2)

        # loop over vals
        data_dict = {}
        for val in split_vals:
            df_filter = df[df[split]==val][col]
            data_dict.update({val: f'{df_filter.mean().round(2)} ({ df_filter.std().round(2)})'})
        
        data_dict.update({'Overall': f'{round(mean_overall, 1)} ({round(std_overall, 1)})',
                    'Demographics': col_new
                    })
        
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
    