"""
ABCD data access utility

Defines access methods to directories relevant to the ABCD Study.
This version allows for flexible data directory configuration:
- Uses an environment variable (ABCD_DATA_DIR) if set.
- Otherwise, falls back to a default local path relative to this script.
"""

from pathlib import Path
import os


# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent


# === Determine the base data directory ===

# Option 1: use environment variable if user has it set
DATA_ROOT = os.getenv("ABCD_DATA_DIR")

# Option 2: fall back to a default relative path (customize for your setup)
if DATA_ROOT is None:
    DATA_ROOT = BASE_DIR / "../../../../../ABCD/om2_replica/data/abcd/release_5.1/core"
else:
    DATA_ROOT = Path(DATA_ROOT).expanduser().resolve()


# === Directories ===

def get_mental_health_directory():
    """Return the path to the mental health data directory."""
    return (DATA_ROOT / "mental-health").resolve()

def get_general_directory():
    """Return the path to the general data directory."""
    return (DATA_ROOT / "abcd-general").resolve()

def get_imaging_directory():
    """Return the path to the imaging data directory."""
    return (DATA_ROOT / "imaging").resolve()

def get_physical_health_directory():
    """Return the path to the physical health data directory."""
    return (DATA_ROOT / "physical-health").resolve()

def get_culture_environment_directory():
    """Return the path to the culture & environment data directory."""
    return (DATA_ROOT / "culture-environment").resolve()

def get_substance_use_directory():
    """Return the path to the substance use data directory."""
    return (DATA_ROOT / "substance-use").resolve()

def get_neurocognition_directory():
    """Return the path to the neurocognition data directory."""
    return (DATA_ROOT / "neurocognition").resolve()


# === General Files ===

def get_demo_file():
    """Path to demographics file."""
    return (get_general_directory() / "abcd_p_demo.csv").resolve()

def get_longitudinal_file():
    """Path to longitudinal tracking file."""
    return (get_general_directory() / "abcd_y_lt.csv").resolve()


# === Mental Health Files ===

def get_ksads_youth_file():
    return (get_mental_health_directory() / "mh_y_ksads_ss.csv").resolve()

def get_ksads_parent_file():
    return (get_mental_health_directory() / "mh_p_ksads_ss.csv").resolve()

def get_ksads_background_parent_file():
    return (get_mental_health_directory() / "mh_p_ksads_bg.csv").resolve()

def get_ksads_background_youth_file():
    return (get_mental_health_directory() / "mh_y_ksads_bg.csv").resolve()

def get_ksads_ptsd_parent_file():
    return (get_mental_health_directory() / "mh_p_ksads_ptsd.csv").resolve()

def get_ksads_suicidality_youth_file():
    return (get_mental_health_directory() / "mh_y_ksads_si.csv").resolve()

def get_ksads_suicidality_parent_file():
    return (get_mental_health_directory() / "mh_p_ksads_si.csv").resolve()

def get_ksads_psychosis_parent_file():
    return (get_mental_health_directory() / "mh_p_ksads_psy.csv").resolve()

def get_cbcl_parent_file():
    return (get_mental_health_directory() / "mh_p_cbcl.csv").resolve()

def get_bpm_youth_file():
    return (get_mental_health_directory() / "mh_y_bpm.csv").resolve()

def get_asr_parent_file():
    return (get_mental_health_directory() / "mh_p_asr.csv").resolve()

def get_abcl_parent_file():
    return (get_mental_health_directory() / "mh_p_abcl.csv").resolve()

def get_bpm_teacher_file():
    return (get_mental_health_directory() / "mh_t_bpm.csv").resolve()

def get_pps_youth_file():
    return (get_mental_health_directory() / "mh_y_pps.csv").resolve()

def get_family_history_parent_file():
    return (get_mental_health_directory() / "mh_p_fhx.csv").resolve()

def get_other_resilience_youth_file():
    return (get_mental_health_directory() / "mh_y_or.csv").resolve()

def get_peer_experience_youth_file():
    return (get_mental_health_directory() / "mh_y_peq.csv").resolve()

def get_cyberbullying_youth_file():
    return (get_mental_health_directory() / "mh_y_cbb.csv").resolve()

def get_adverse_life_events_youth_file():
    return (get_mental_health_directory() / "mh_y_le.csv").resolve()

def get_adverse_life_events_parent_file():
    return (get_mental_health_directory() / "mh_p_le.csv").resolve()


# === Imaging Files ===

def get_mri_y_qc_incl_file():
    return (get_imaging_directory() / "mri_y_qc_incl.csv").resolve()


# === Physical Health Files ===

def get_puberty_scale_youth_file():
    return (get_physical_health_directory() / "ph_y_pds.csv").resolve()

def get_developmental_history_parent_file():
    return (get_physical_health_directory() / "ph_p_dhx.csv").resolve()

def get_medications_inventory_parent_file():
    return (get_physical_health_directory() / "ph_p_meds.csv").resolve()

# === Culture and Environment ===

def get_parental_monitoring_youth_file():
    return (get_culture_environment_directory() / "ce_y_pm.csv").resolve()

def get_parental_behavior_youth_file():
    return (get_culture_environment_directory() / "ce_y_crpbi.csv").resolve()

def get_discrimination_scale_youth_file():
    return (get_culture_environment_directory() / "ce_y_dm.csv").resolve()

def get_neighborhood_safety_crime_parent_file():
    return (get_culture_environment_directory() / "ce_p_nsc.csv").resolve()

def get_neighborhood_safety_crime_youth_file():
    return (get_culture_environment_directory() / "ce_y_nsc.csv").resolve()

def get_school_attendance_grades_parent_file():
    return (get_culture_environment_directory() / "ce_p_sag.csv").resolve()

def get_school_attendance_grades_youth_file():
    return (get_culture_environment_directory() / "ce_y_sag.csv").resolve()


# === Substance Use Files ===

def get_substance_use_interview_youth_file():
    return (get_substance_use_directory() / "su_y_sui.csv").resolve()

def get_substance_use_timeline_youth_file():
    return (get_substance_use_directory() / "su_y_tlfb.csv").resolve()

def get_substance_use_interview_mid_year_youth_file():
    return (get_substance_use_directory() / "su_y_mypi.csv").resolve()


# === Neurocognition Files ===

def get_nih_toolbox_youth_file():
    return (get_neurocognition_directory() / "nc_y_nihtb.csv").resolve()

def get_ravlt_youth_file():
    return (get_neurocognition_directory() / "nc_y_ravlt.csv").resolve()

def get_lmt_youth_file():
    return (get_neurocognition_directory() / "nc_y_lmt.csv").resolve()

def get_wisc_youth_file():
    return (get_neurocognition_directory() / "nc_y_wisc.csv").resolve()

