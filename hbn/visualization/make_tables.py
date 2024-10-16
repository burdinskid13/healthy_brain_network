def remap_cols(vals):
    map = {'female': 'Female sex assigned at birth, n (%)',
            'PreInt_Demos_Fam,Child_Race_cat': 'Race',
            'White/Caucasian': 'White race, n (%)',
            'Hispanic': 'Hispanic ethnicity, n (%)',
            'Black/African American': 'Black race, n (%)',
            'Unknown': 'Unknown, n (%)',
            'Two or more races': 'Mixed race, n (%)',
            'Asian': 'Asian race, n (%)',
            'Age_round': 'Age, years, mean (SD)',
            'comorbidities': 'Comorbidities, mean (SD)',
            'Above NYC poverty line': 'Household, Above NYC poverty line, n (%)',
            'College Degree or Higher': 'Parental Education, College Degree or Higher, n (%)',
            }
    # programmatically reformat diagnosis
    data_dict = {}
    for val in vals:
        if val in map.keys():
            data_dict.update({val: map[val]})
        else:
            data_dict.update({val: f'{val}, n (%)'})

    return data_dict


def remap_ses():
    return {
        0: "<$10,000",
        1: "$10,000 - $19,999",
        2: "$20,000 - $29,999",
        3: "$30,000 - $39,999",
        4: "$40,000 - $49,999",
        5: "$50,000 - $59,999",
        6: "$60,000 - $69,999",
        7: "$70,000 - $79,999",
        8: "$80,000 - $89,999",
        9: "$90,000 - $99,999",
        10: "$100,000 - $149,999",
        11: "$150,000 or more"
        }


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

    out_dict = {'All Groups': f'{round(all_count, 1)} ({round(all_perc, 1)})'}
    for name, count, perc in zip(vals, counts, percs):
        out_dict.update({f'{name}': f'{round(count, 1)} ({round(perc, 1)})'})

    return out_dict


def overall_numbers(df, col, vals=['No Diagnosis Given', 'Reading Impairment']):
    # get overall numbers for `overall`
    # loop over vals
    counts = []
    for val in vals:
        counts.append(sum(df[col]==val))

    out_list = [sum(counts)] + counts

    return out_list


def make_table_count(df,
                    cols,
                    vals,
                    split='Diagnosis',
                    split_vals=['No Diagnosis Given', 'Reading Impairment'],
                    overall=None
                    ):
    from collections import defaultdict
    import pandas as pd

    if 'Identifiers' in df.columns:
        # get patient count
        df_count = df.groupby('Identifiers').head(1)
    else:
        df_count = df

    # get overall count
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
    df_out['Demographics'] = df_out['Demographics'].map(remap_cols(vals))
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
        
        data_dict.update({'All Groups': f'{round(mean_overall, 1)} ({round(std_overall, 1)})',
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


def add_SES(df):
    import os
    from hbn.constants import Defaults
    import pandas as pd
    
    # cols to keep
    cols_to_keep = ['Identifiers', 'FSQ,FSQ_04', 'Barratt,Barratt_Total_Edu', 'Barratt,Barratt_Total']

    # load barratt
    df_SES = pd.read_csv(os.path.join(Defaults.INTERIM_FEATURES_DIR, 'Parent-features-raw.csv'), engine='python')[cols_to_keep]

    # make some new cols for income
    df_SES['household_income'] = df_SES['FSQ,FSQ_04'].map(remap_ses())
    df_SES.loc[df_SES['FSQ,FSQ_04']<=4, 'poverty_line'] = 'Below NYC poverty line'
    df_SES.loc[df_SES['FSQ,FSQ_04']>4, 'poverty_line'] = 'Above NYC poverty line'

    # make new cols for education
    df_SES.loc[df_SES['Barratt,Barratt_Total_Edu']>=18, 'education'] = 'College Degree or Higher'
    df_SES.loc[df_SES['Barratt,Barratt_Total_Edu']<18, 'education'] = 'Less than College Degree'

    # merge with dataframe
    df_out = df.merge(df_SES, on='Identifiers', how='outer')

    return df_out


def make_table_1(split='DX_Reading', split_vals=['No Diagnosis Given', 'reading_all_comorbidities', 'reading_no_comorbidities']):
    """make table 1
    """
    import pandas as pd
    import os
    from hbn.constants import Defaults
    from hbn.visualization import utils

    # load data
    df = pd.read_csv(os.path.join(Defaults.INTERIM_FEATURES_DIR, 'participant_train_test.csv'))

    # rename cols
    df = df.rename(columns={'PreInt_Demos_Fam,Child_Race_cat': 'Race'})

    # remap diagnosis
    df[split] = df[split].map(utils.remap_dx())

    # mean and std for age and comorbd.
    df_mean_std = make_table_mean_std(df=df,
                                    cols=['Age_round', 'comorbidities'],
                                    cols_new=['Age, years, mean (SD)', 'Comorbidities, mean (SD)'],
                                    split=split,
                                    split_vals=split_vals
                                    )
    
    # add SES
    df = add_SES(df)
    
    # make table count
    df_count = make_table_count(df=df, 
                                cols=['Race', 'Race','Race', 'Race', 'Race', 'Race', 'Sex', 'poverty_line', 'education'], 
                                vals=['Unknown', 'Hispanic', 'Two or more races', 
                                      'White/Caucasian', 'Black/African American', 'Asian', 'female', 'Above NYC poverty line', 'College Degree or Higher'],
                                split=split,
                                split_vals=split_vals,
                                )
    # concat dataframes
    desired_row_order = ['n', 'Age, years, mean (SD)', 'Female sex assigned at birth, n (%)', 'Household, Above NYC poverty line, n (%)', 'Parental Education, College Degree or Higher, n (%)',
                 'Comorbidities, mean (SD)',  'White race, n (%)', 'Black race, n (%)', 'Mixed race, n (%)',
                 'Asian race, n (%)', 'Hispanic ethnicity, n (%)', 'Unknown, n (%)'
                ]
    df_concat = pd.concat([df_mean_std, df_count]).set_index('Demographics')
    df_full = df_concat.iloc[df_concat.index.get_indexer(desired_row_order)]
    
    make_table_png(df=df_full, outpath=os.path.join(Defaults.FIG_DIR, 'table1.png'))


def make_table_2():
    from hbn.visualization import utils
    import pandas as pd
    import os
    from hbn.constants import Defaults
    from matplotlib import pyplot as plt

    # load all participant diagnoses
    df = pd.read_csv(os.path.join(Defaults.INTERIM_FEATURES_DIR, 'all_participant_diagnoses.csv'))

    df.rename(columns={'DX_Cat_Name': 'Diagnosis'}, inplace=True)

    # add comorbidities
    dx_counts = df.groupby('Identifiers')['Diagnosis'].apply(lambda x: x.nunique()-1).reset_index(name='comorbidities')
    df = dx_counts.merge(df, on='Identifiers')

    df1 = df.groupby('Diagnosis')[['Diagnosis']].value_counts().reset_index(name='Overall Count')

    # calculate comoridities
    df2 = df.groupby('Diagnosis')['comorbidities'].agg({'mean', 'std'}).reset_index()
    df2['Comorbidities, mean (SD)'] = df2['mean'].round(2).astype(str) + ' (' + df2['std'].round(2).astype(str) + ')'

    # calculate the percentage of females
    df3 = df.groupby('Diagnosis')['Sex'].apply(lambda x: (x == 'female').sum()).reset_index(name='n of females')
    df4 = df.groupby('Diagnosis')['Sex'].apply(lambda x: (x == 'female').mean() *100).reset_index(name='percentage of females')
    df4['Female sex assigned at birth, n (%)'] = df3['n of females'].astype(str) + ' (' + df4['percentage of females'].round(2).astype(str) + ')'

    # calculate race
    df5 = df.groupby('Diagnosis')['PreInt_Demos_Fam,Child_Race_cat'].apply(lambda x: (x == 'White/Caucasian').sum()).reset_index(name='n of White/Caucasian')
    df6 = df.groupby('Diagnosis')['PreInt_Demos_Fam,Child_Race_cat'].apply(lambda x: (x == 'White/Caucasian').mean() *100).reset_index(name='percentage of White/Caucasian')
    df6['White Race, n (%)'] = df5['n of White/Caucasian'].astype(str) + ' (' + df6['percentage of White/Caucasian'].round(2).astype(str) + ')'

    # merge dataframes
    df_merged = df1.merge(df2[['Diagnosis', 'Comorbidities, mean (SD)']], on='Diagnosis') \
            .merge(df4[['Diagnosis', 'Female sex assigned at birth, n (%)']], on='Diagnosis') \
            .merge(df6[['Diagnosis', 'White Race, n (%)']], on='Diagnosis')

    df_merged = df_merged.sort_values(by='Overall Count', ascending=False).set_index('Diagnosis').head(20)

    make_table_png(df=df_merged, outpath=os.path.join(Defaults.FIG_DIR, 'table2.png'))


def make_table_3():
    pass
    