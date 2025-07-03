def remap_cols(vals):
    map = {'female': 'Female sex assigned at birth, n (%)',
            'White/Caucasian': 'White race, n (%)',
            'Hispanic': 'Hispanic ethnicity, n (%)',
            'Black/African American': 'Black race, n (%)',
            'Unknown': 'Unknown, n (%)',
            'Two or more races': 'Mixed race, n (%)',
            'Asian': 'Asian race, n (%)',
            'age_round': 'Age, years, mean (SD)',
            'comorbidities_DX_Subtype': 'Comorbidities, mean (SD)',
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


def count_perc_vals(val_counts, overall_counts, val_names=['No Diagnosis Given', 'Reading Impairment']):

    # get overall count
    all_count = sum(val_counts)
    all_perc = all_count / (sum(overall_counts))*100

    out_dict = {'All Groups': f'{round(all_count, 1)} ({round(all_perc, 1)})'}
    for name, count, overall in zip(val_names, val_counts, overall_counts):
        perc = count / (overall)*100
        out_dict.update({f'{name}': f'{round(count, 1)} ({round(perc, 1)})'})

    return out_dict


def overall_numbers(df, col, vals=['No Diagnosis Given', 'Reading Impairment']):
    # get overall numbers for `overall`
    # loop over vals
    counts = []
    for val in vals:
        participants = df[df[col]==val]['Identifiers'].unique()
        counts.append(len(participants))

    return counts


def make_table_count(df,
                    cols,
                    vals,
                    split='Diagnosis',
                    split_vals=['No Diagnosis Given', 'Reading Impairment'],
                    overall=None
                    ):
    from collections import defaultdict
    import pandas as pd

    # get overall count
    overall_counts = overall_numbers(df=df, col=split, vals=split_vals)

    # loop over cols + corresponding vals
    out_dict = defaultdict(list)
    for (col, val) in zip(cols, vals): 

        # filter dataframe
        df_filter = df[df[col]==val].reset_index(drop=True)

        # get overall count for filtered dataframe
        val_counts = overall_numbers(df=df_filter, col=split, vals=split_vals)

        # get patient count and % of overall
        data_dict = count_perc_vals(
                                    val_counts=val_counts,
                                    overall_counts=overall_counts,
                                    val_names=split_vals
                                    )
        
        for k,v in data_dict.items():
            out_dict[k].append(v)
        
    # make dataframe and clean up
    df_out = pd.DataFrame.from_dict(out_dict)
    df_out['Demographics'] = vals
    df_out['Demographics'] = df_out['Demographics'].map(remap_cols(vals))

    all_counts = [sum(overall_counts)] + overall_counts
    first_row = pd.DataFrame([pd.Series(all_counts + ['n'], index=df_out.columns)])
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
    import textwrap

    fig, ax = plt.subplots()
    ax.axis('off')

    table = pd.plotting.table(ax, df, loc='center', cellLoc='center')

    # --- Customize Font Size ---
    table.auto_set_font_size(False)  # Disable auto font size
    table.set_fontsize(10)           # Set a larger font size (adjust as needed)

    # --- Customize Cell Properties for Clarity (Optional but Recommended) ---
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)      # Add cell borders for better separation (adjust as needed)
        cell.set_edgecolor('black')  # Set border color (optional)
        cell.set_text_props(ha='center', va='center') # Ensure text is centered (redundant if cellLoc='center')

    # --- Customize First Row: Remove Borders and Adjust Text Position ---
    for i in range(df.shape[1]):  # Iterate through the columns (first row)
        cell = table[0, i]  # Access the cell at row 0, column i

        # Remove top and bottom spines (horizontal lines)
        cell.visible_edges = 'LR'  # Keep only left and right edges

        # Remove left and right spines (vertical lines)
        cell.set_edgecolor('white') # Set edge color to white to hide them

        # Adjust text properties for better visual separation
        text = cell.get_text()
        text.set_ha('center')  # Center horizontally (already set globally)
        text.set_va('bottom')  # Align text to the bottom of the (now borderless) area

    # --- Customize Subsequent Rows (Optional - if you want borders for the data) ---
    for i in range(1, df.shape[0] + 1):  # Iterate through data rows (starting from index 1)
        for j in range(df.shape[1]):
            cell = table[i, j]
            cell.set_linewidth(0.5)
            cell.set_edgecolor('black')

        # --- Customize First Row: Rotate Text ---
    for i in range(df.shape[1]):  # Iterate through the columns (first row)
        cell = table[0, i]  # Access the cell at row 0, column i
        text = cell.get_text()
        text._text = textwrap.fill(text._text, width=5, break_long_words=False)
        # text.set_rotation(45)
        # text.set_ha('left')  # Adjust horizontal alignment for rotated text
        # text.set_va('center') # Adjust vertical alignment for rotated text


    plt.savefig(outpath)


def remap_SES(df):
    import pandas as pd

    mapping_dict = {
        "<$10,000": 0,
        "$10,000 - $19,999": 1,
        "$20,000 - $29,999": 2,
        "$30,000 - $39,999": 3,
        "$40,000 - $49,999": 4,
        "$50,000 - $59,999": 5,
        "$60,000 - $69,999": 6,
        "$70,000 - $79,999": 7,
        "$80,000 - $89,999": 8,
        "$90,000 - $99,999": 9,
        "$100,000 - $149,999": 10,
        "$150,000 or more": 11
        }

    # make some new cols for income
    df['household_income'] = df['household_income'].map(mapping_dict)
    df.loc[df['household_income']<=4, 'poverty_line'] = 'Below NYC poverty line'
    df.loc[df['household_income']>4, 'poverty_line'] = 'Above NYC poverty line'

    return df


def make_table_1(
        df,
        split='DX_Reading', 
        split_vals=['No Diagnosis Given', 'reading_all_comorbidities', 'reading_no_comorbidities'],
        split_val_new_names=None
        ):
    """make table 1
    """
    import pandas as pd
    import os
    from hbn.constants import Defaults
    from hbn.visualization import utils

    # mean and std for age and comorbd.
    df_mean_std = make_table_mean_std(df=df,
                                    cols=['age_round', 'comorbidities_DX_Subtype'],
                                    cols_new=['Age, years, mean (SD)', 'Comorbidities, mean (SD)'],
                                    split=split,
                                    split_vals=split_vals
                                    )
    
    # make table count
    df = remap_SES(df)

    df_count = make_table_count(df=df, 
                                cols=['race', 'race','race', 'race', 'race', 'race', 'sex', 'poverty_line', 'education'], 
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

    # remap diagnosis
    if split_val_new_names is not None:
        df_full.columns = split_val_new_names
        

    make_table_png(df=df_full, outpath=os.path.join(Defaults.FIG_DIR, 'table1.png'))
