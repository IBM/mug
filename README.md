# Model of Urban Growth

![](https://github.com/IBM/mug/blob/867ab0498df9dd85be74ef8acad9c264478f8f78/assets/ug.gif)


This repository contains code to mount a dataset which connects demographic and geospatial data using [`torchgeo`](https://github.com/microsoft/torchgeo) and a deep learning model with sequence-to-sequence convLSTM architecture with training, validation and testing for urban settlements prediction. The model training used a time series from [Worldpop.org](https://www.worldpop.org/) between 2000 and 2004 and the test carried out used datafrom the year 2015.


## Download data

The data can be downloaded into a new directory `**/worldpop/**` . For this test, we are using land cover, water, roads and slope information. The connection between different datasets will performed by the code in [`src/mug/dataset/*`](https://github.com/IBM/mug/blob/main/src/mug/datasets) using Torchgeo.

* [Age & Sex structures](https://hub.worldpop.org/project/categories?id=8)
* [Elevation](https://hub.worldpop.org/geodata/listing?id=58)
* [Population](https://hub.worldpop.org/project/categories?id=3)
* [LandCover](https://hub.worldpop.org/geodata/listing?id=60)

These data files are downloaded using FTP:

* Slope: ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/Slope/slope100m.tif
* Water: ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/OSM_Water/BinaryMosaic_1_0_NoData/osmwater100m_1_0_NoData.tif
* Roads: ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/OSM_Roads/BinaryMosaic_1_0_NoData/osmhighway100m8-1710nd.tif

More details can be found in [`src/mug/datasets/worldpop.py`](https://github.com/IBM/mug/blob/main/src/mug/datasets/worldpop.py).


## Installation

Clone this repository.
```shell
$ git clone git@github.com:IBM/mug.git
$ export MUG_HOME='./mug'
```
### Creating the virtual environment

First, you will need to install Python, version 3.12.2. The following instructions assume you are already in the `mug` repo root directory.

1. Create the env: `python3 -m venv venv`
2. Activate the env: `source ./venv/bin/activate`
3. Install all packages: `pip install -r ./requirements.txt`
4. Install `mug` in editable mode:`pip install -e .`

Here are the commands as one script:

```shell
python3 -m venv venv
source ./venv/bin/activate
pip install -r ./requirements.txt
pip install -e .
```

## Usage

There are two ways of run the code: by notebook or by script

### Notebook

> [!NOTE]
> You will have to activate the virtual environment everytime you need to use `mug`. 

For instance, if you want to work with `mug` on `jupyter`, you will need to run the following commands in a freshly started shell:
```shell
source ./venv/bin/activate
jupyter-notebook trainjupyter_s2s_main.ipynb
```

### Python script

> [!NOTE]
> You will have to activate the virtual environment everytime you need to use `mug`. 

For instance, if you want to work with `mug` using a Python script file you will need to run the following commands in a freshly started shell:
```shell
source ./venv/bin/activate
python script_mug_s2s.py --parameter_file='parameters_variable.yaml'
```


### File Structure

```
src/mug/            
├─ cli/                     - utilities regarding an alternative code to load data and model 
├─ dataset/                 - dataset management (download, processing, generation )
├── nn/                     - model architecture
├─ samples/                 - sample generation
└─ utils/                   - utilities regarding evaluation

requirements.txt            - Python library requirements
rois.csv                    - config file with region of interest using latitude and longitude of each city      
parameters_variable.yaml    - parameter file with hyperparameters
script_mug_s2s.py           - main script to load the data, train, valid and test the sequence-to-sequence convLSTM model 
trainjupyter_s2s_main.ipynb - main notebook code to load the data, train, valid and test the sequence-to-sequence convLSTM model 
```

## Results (test images)

![](https://github.com/IBM/mug/blob/9dbae9c86e29799dacedb62fda38feee016ad661/assets/result_graphic.png)
