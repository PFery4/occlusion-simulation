# OCCLUSION SIMULATOR FOR TRAJECTORY DATASETS
This repository contains the code for a simulator, that is capable of generating *spatio-temporally defined occlusions* from trajectory data that is widely available within the trajectory prediction research field.

Our simulator generates occlusion cases from agents' trajectories, focusing on generating cases for which the agents are *currently occluded*.

We provide the code for our simulator, as well as the necessary functionalities for its operation on the [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/).

## Installation / Setup
### Environment

*We recommend to use Anaconda to host the project's environment: the project relies on the scikit-geometry library, which is directly accessible from the conda-forge channel.
Though installation through other methods might be possible, only the following instructions have been verified to work properly.*

1. Create the environment:
   ```
   conda create -n <environment-name> python=3.8 pip
   ```
   Replace `<environment-name>` with the name of your environment.
2. Activate the environment:
    ```
    conda activate <environment-name>
    ```
3. Install [PyTorch 1.8.0](https://pytorch.org/get-started/previous-versions/#v180) with the appropriate CUDA version.
4. Install scikit-geometry:
   ```
   conda install -c conda-forge scikit-geometry
   ```
5. Install the remaining dependencies:
   ```
   pip install -r requirements.txt
   ```

### Stanford Drone Dataset

1. Download the [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) and extract its contents anywhere on your system.
2. Inside the [config/SDD_dataset_config.yaml](config/SDD_dataset_config.yaml) file, fill the `path` entry with the root directory of your extracted Stanford Drone Dataset (i.e., the directory where the dataset's README and the annotation directory are located).

### Environment Variables

1. Add this directory to the PYTHONPATH environment variable:
   ```
   export PYTHONPATH=$PWD
   ```
   
### Coordinates Conversion File

1. Create the coordinates conversion `.txt` file inside the [config](config) directory by running the [src/data/save_coord_conv_file.py](src/data/save_coord_conv_file.py) script:
   ```
   python src/data/save_coord_conv_file.py
   ```

## Dataset Processing

### Preprocessing

Before running the simulator, the dataset must first be preprocessed (the preprocessing involves subsampling of measurements to a desired frequency, removal of undesirable trajectories and selection of desired agent classes).
The preprocessing of the dataset can be parametrized using the [config/dataset_preprocessing_config.yaml](config/dataset_preprocessing_config.yaml) file. Feel free to adapt the parameters to your liking. When ready, you can run the preprocessing of the dataset by running the [src/data/dataset_saving.py](src/data/dataset_saving.py) script, with the following command:
```
python src/data/dataset_saving.py --preprocessing-cfg dataset_preprocessing_config --dir-name SDD_base
```
This will create a directory `outputs/pickled_dataloaders/SDD_base`, where the preprocessed dataset will be stored.
