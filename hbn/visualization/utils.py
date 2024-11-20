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
                'ADHD-Combined Type (no Depression)_Depression (no ADHD)': 'ADHD (Combined Type) (no Depression)',
                'ADHD-Inattentive Type (no Depression)_Depression (no ADHD)': 'ADHD (Inattentive Type) (no Depression)',
                'ADHD-Combined Type (no Anxiety)_Anxiety (no ADHD)': 'ADHD (Combined Type) (no Anxiety)',
                'ADHD-Inattentive Type (no Anxiety)_Anxiety (no ADHD)': 'ADHD (Inattentive Type) (no Anxiety)',
                'ADHD-Inattentive Type (no Depression)': 'ADHD (Inattentive Type) (no Depression)',
                'ADHD-Combined Type (no Depression)': 'ADHD (Combined Type) (no Depression)',
                'ADHD-Inattentive Type (no Anxiety)': 'ADHD (Inattentive Type) (no Anxiety)',
                'ADHD-Combined Type (no Anxiety)': 'ADHD (Combined Type) (no Anxiety)',
                'Depression (no ADHD)': 'Depression (no ADHD)',
                'Anxiety (no ADHD)': 'Anxiety (no ADHD)',
                'Disruptive, Impulse Control and Conduct Disorders': 'Conduct Disorders',
                'Specific Learning Disorder with Impairment in Reading': 'Reading Impairment',
                'Obsessive Compulsive and Related Disorders': 'Obsessive Compulsive Disorder',
                }
    
    return data_dict
