# The scripts used for stirng vessel project

# How to install

* step 1: create a new [conda](https://docs.anaconda.com/free/anaconda/install/windows/) environment and activate it

```bash
conda create -y -n stringvessel python==3.11
conda activate stringvessel
```

* step 2: intall necessary packages:

# install vessel analysis package
pip install git+https://github.com/MMV-Lab/vessel_analysis_3d.git

# clone this repo and install other dependencies

```
git clone https://github.com/MMV-Lab/string_vessel_project.git
cd string_vessel_project
pip install -r requirements.txt 
```

## How to use

Follow the in-line instructions in each jupyter notebook. Here, we just provide a general overview.

Download trained models from this [LINK](https://ambiomcloud.isas.de/index.php/s/CwcfFRt8eQ9gKWj)

### part 1: 

    data_wrangling.ipynb  This script is to convert the original Leica files into proper tiff files. 

For some datasets, loading the original Leica files with aicsimageio may mess up the dimensions, i.e., C and Z are swapped.
So, for some datasets, we may need to re-order the dimensions to make sure we have images of shape CZYX (C=2, Z=7)


### part 2:

    inference_2class.ipynb will apply the 2D model on 3D data.

<b>NOTE:</b> This notebook now contain some helpful instructions for testing the newly provided models.

This step will generate a segmentation for each file. In segmentation, string vessels will have pixel value 2, and normal vessel will have pixel value 1, while background will have
pixel value 0. 

### part 3:

    vessel_analysis.ipynb will extract quantitative features from the segmentation results


