from hbn.constants import Defaults
import os

def make_specs(out_dir=os.path.join(Defaults.MODEL_SPEC_DIR, 'participant_specs')):
    import os
    from hbn import io

    spec_info = {
        'spec-adhd-No_Diagnosis_01':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis_01':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-Subtypes_02':
        {'diagnoses': ['ADHD-Combined_Type', 'ADHD-Inattentive_Type'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-depression-No_Diagnosis_01':
        {'diagnoses': ['Depressive_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-anxiety-No_Diagnosis_01':
        {'diagnoses': ['Anxiety_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-reading-No_Diagnosis_01':
        {'diagnoses': ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-No_Diagnosis_03':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-No_Diagnosis_04':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis_02':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis_03':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-reading-No_Diagnosis_02':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-No_Diagnosis_03':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-No_Diagnosis_02':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-No_Diagnosis_03':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-depression-No_Diagnosis_02':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-depression-No_Diagnosis_03':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-age-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-17':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-18':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [18,19,20,21,22],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-17':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-18':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [18,19,20,21,22],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-17':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-18':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [18,19,20,21,22],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-anxiety-05':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-depression-06':
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
        'spec-adhd-asd-07':
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
        'spec-adhd-reading-08':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-anxiety-09':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-depression-10':
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
        'spec-adhd-asd-11':
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
        'spec-adhd-reading-12':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-anxiety-13':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-depression-14':
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
        'spec-adhd-asd-15':
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
        'spec-adhd-reading-16':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
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
            "Anxiety_Disorders",
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

    