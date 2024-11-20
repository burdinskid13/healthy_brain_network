import os
import pandas as pd # to read/manipulate/write data from files
import numpy as np # to manipulate data/generate random numbers
import plotly.express as px # interactive visualizations
import seaborn as sns # static visualizations
import matplotlib.pyplot as plt # fine tune control over visualizations

from pathlib import Path # represent and interact with directories/folders in the operating system
from collections import namedtuple # structure data in an easy to consume way

import requests # retrieve data from an online source

def get_elements(files, in_dir, out_dir):
    # We store the info in 4 different Python datatypes
    data_elements = []
    data_structures = {}
    event_names = set()
    StructureInfo = namedtuple("StructureInfo", field_names=["description", "eventnames"])

    for text_file in files:
        # Extract data structure from filename
        data_structure = Path(text_file).name.split('.txt')[0]
        
        # Read the data structure and capture all the elements from the file
        # Note this could have been done using the data returned from the NDA API
        # We are using pandas to read both the first and second rows of the file as the header
        # Note: by convention dataframe variables contain `df` in the name.
        data_structure_df = pd.read_table(text_file, header=[0, 1], nrows=0)
        for data_element, metadata in data_structure_df.columns.values.tolist():
            data_elements.append([data_element, metadata, data_structure])

        
        # (Optional) Retrieve the eventnames in each structure. Some structures were only collected
        # at baseline while others were collected at specific or multiple timepoints
        events_in_structure = None
        if any(['eventname' == data_element for data_element in data_structure_df.columns.levels[0]]):
            # Here we are skipping the 2nd row of the file containing description using skiprows
            possible_event_names_df = pd.read_table(text_file, skiprows=[1], usecols=['eventname'])
            events_in_structure = possible_event_names_df.eventname.unique().tolist()
            event_names.update(events_in_structure)

        # (Optional) Retrieve the title for the structure using the NDA API
        rinfo = requests.get(f"https://nda.nih.gov/api/datadictionary/datastructure/{data_structure}").json()
        data_structures[data_structure] = StructureInfo(description=rinfo["title"] if "title" in rinfo else None,
                                                        eventnames=events_in_structure)

    # Convert to a Pandas dataframe
    data_elements_df = pd.DataFrame(data_elements, columns=["element", "description", "structure"])

    # save to csv
    data_elements_df.to_csv("data_elements.tsv", sep="\t", index=None)

def run():
    """
    Script to run the ABCD data elements extraction
    
    Parameters
    ----------
    in_dir : Path
        Location of the ABCD data files
    out_dir : Path
        Where to save the data elements file
    """
    in_dir = Path("/om2/user/maedbh/abcd_data/ABCD4")
    files = files = sorted(in_dir.glob("*.txt"))

    get_elements(files, in_dir=in_dir, out_dir=in_dir)


if __name__ == "__main__":
    run()