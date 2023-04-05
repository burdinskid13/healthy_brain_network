from hbn.constants import Defaults


def make_specs(out_dir=Defaults.MODEL_SPEC_DIR):
    import os
    from hbn import io

    spec_info = {
        'spec1':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec2':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec3':
        {'diagnoses': ['ADHD-Combined_Type', 'ADHD-Inattentive_Type'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec4':
        {'diagnoses': ['Depressive_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec5':
        {'diagnoses': ['Anxiety_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec6':
        {'diagnoses': ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec7':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec8':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec9':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec10':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec11':
        {
        "diagnoses": ["No_Diagnosis_Given"],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec12':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec13':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec14':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec15':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec16':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec17':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec18':
        {
        "diagnoses": ["No_Diagnosis_Given"],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec19':
        {
        "diagnoses": ["No_Diagnosis_Given"],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec20':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec21':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec22':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec23':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec24':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec25':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec26':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec27':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec28':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16,17,18,19,20,21,22],
         'sex': 'all',
         'ethnicity': 'all'
        },
        }

    # loop over classifies and save out pydra-ml specs
    for name, spec in spec_info.items():

        # write out pydra-ml specs
        fpath = os.path.join(out_dir, f'participant_{name}.json')
        io.save_dict_as_JSON(fpath, spec)

    