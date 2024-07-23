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
