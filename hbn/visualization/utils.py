import pandas as pd
import numpy as np
from hbn.features.build_features import _index_dataframe_by_columns_values

def filter_train(df, columns, values):

    return _index_dataframe_by_columns_values(dataframe=df, 
                                            columns=columns, 
                                            values_list=values
                                            )

def remap_dx():
    data_dict = {
                'adhd_all_comorbidities': 'ADHD (all comorbid.)',
                'adhd_no_comorbidities': 'ADHD (no comorbid.)',  
                'other_diagnoses': 'Other diagnoses',  
                'No Diagnosis Given': 'No Diagnosis',
                'ADHD-Combined Type': 'ADHD (Combined Type)',
                'ADHD-Inattentive Type': 'ADHD (Inattentive Type)',
                'ADHD-Hyperactive/Impulsive Type': 'ADHD (Hyperactive Type)',
                'reading_all_comorbidities': 'Reading (all comorbid.)',
                'reading_no_comorbidities': 'Reading (no comorbid.)',
                'adhd_all_comorbidities_No Diagnosis Given': 'ADHD (all comorbid.)',
                'adhd_no_comorbidities_No Diagnosis Given': 'ADHD (no comorbid.)',
                'ADHD-Combined Type_No Diagnosis Given': 'ADHD (Combined Type)',
                'ADHD-Inattentive Type_No Diagnosis Given': 'ADHD (Inattentive Type)',
                }
    
    return data_dict
