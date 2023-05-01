from hbn.constants import Defaults
import os

def make_specs(out_dir=os.path.join(Defaults.MODEL_SPEC_DIR, 'participant_specs')):
    import os
    from hbn import io

    spec_info = {
        'spec-adhd-01':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-asd-01':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-02':
        {'diagnoses': ['ADHD-Combined_Type', 'ADHD-Inattentive_Type'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-depression-01':
        {'diagnoses': ['Depressive_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-anxiety-01':
        {'diagnoses': ['Anxiety_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-reading-01':
        {'diagnoses': ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-03':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-04':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-asd-02':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-asd-03':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-reading-02':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-03':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-02':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-03':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-depression-02':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-depression-03':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16,17,18,19,20,21,22],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-17':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-18':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-19':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-20':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-21':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-22':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16,17,18,19,20,21,22],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-23':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-24':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-25':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-26':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-27':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-28':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-29':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-30':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-31':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16,17,18,19,20,21,22],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-32':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-33':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-34':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-35':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-36':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-37':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-38':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-39':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-40':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-41':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-42':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-43':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-multilabel':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder",
            "Anxiety",
            "Depressive_Disorders",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        }

    # loop over classifies and save out specs
    io.make_dirs(out_dir)
    for name, spec in spec_info.items():

        # write out participant specs
        fpath = os.path.join(out_dir, f'{name}.json')
        io.save_dict_as_JSON(fpath, spec)

    