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
        }

    # loop over classifies and save out pydra-ml specs
    for name, spec in spec_info.items():

        # write out pydra-ml specs
        fpath = os.path.join(out_dir, f'participant_{name}.json')
        io.save_dict_as_JSON(fpath, spec)

    