# Data Documentation

## Dataset Source

This project uses the DANDI archive dataset 000138:

- **DANDI ID**: 000138
- **Subject**: Macaque monkey (Jenkins)
- **Session**: Large dataset
- **File**: `sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb`

## Data Description

### Neural Data
- **Recording Type**: Extracellular spikes
- **Brain Area**: Motor cortex
- **Number of Units**: Variable (typically 50-100)
- **Sampling**: Spike timestamps

### Behavioral Data
- **Modality**: Hand kinematics
- **Variables**: X and Y velocity (cm/s)
- **Sampling Rate**: Variable (~100-200 Hz)
- **Task**: Center-out reaching task

### Trial Structure
- **Number of Trials**: ~100-200
- **Trial Duration**: ~5-10 seconds
- **Start/Stop Times**: Provided in NWB trials table

## Data Access

