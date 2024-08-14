# Model of Urban Growth

![](https://github.com/IBM/mug/blob/f2e701dcd4bddc056ed4287c59df6f02b79c0cec/assets/ug.gif)


This repository contains code to mount a dataset which connects demographic and geospatial data using **torchgeo** and a deep learning model with sequence-to-sequence convLSTM architecture with training, validation and testing for urban settlements prediction. The model training used a time series from **Worldpop.org** between 2000 and 2004 and the test carried out used datafrom the year 2015.


## Download data

[] [Age & Sex structures](https://hub.worldpop.org/project/categories?id=8)

[] [Elevation](https://hub.worldpop.org/geodata/listing?id=58)

[] [Population](https://hub.worldpop.org/project/categories?id=3)

[] [LandCover](https://hub.worldpop.org/geodata/listing?id=60)

[] Slope 

FTP protocol:  

ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/Slope/slope100m.tif


[] Water

FTP protocol: 

ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/OSM_Water/BinaryMosaic_1_0_NoData/osmwater100m_1_0_NoData.tif

[] Roads

FTP protocol:

ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/OSM_Roads/BinaryMosaic_1_0_NoData/osmhighway100m8-1710nd.tif



More details you can find at /src/mug/worldpop.py



## Installation

Clone this repository.
```shell
$ git clone git@github.com:IBM/mug.git
$ export MUG_HOME='./mug'
```
### Creating the virtual environment

First, you will need to install Python, version 3.12.2.

1. Create the env: `python3 -m venv venv`
2. Activate the env: `source ./venv/bin/activate`
3. Install all packages: `(venv)$ pip install -r ${MUG_HOME}/requirements.txt`
4. Install `mug` in editable mode.:`(venv)$ pip install -e ${MUG_HOME}`


## Usage

There are two ways of run the code: by notebook or by script

### Notebook
You will have to activate the virtual environment everytime you need to use `mug`. 

For instance, if you want to work with `mug` on `jupyter`, you will need to run the following commands in a freshly started shell:
```shell
$ source ./venv/bin/activate
$ cd mug
(venv)$ jupyter-notebook trainjupyter_s2s_main.ipynb
```

### Python script

You will have to activate the virtual environment everytime you need to use `mug`. For instance, if you want to work with `mug` using a Python script file you will need to run the following commands in a freshly started shell:
```shell
$ source ./venv/bin/activate
$ cd mug
(venv)$ python script_mug_s2s.py --parameter_file='parameters_variable.yaml'
```


### File Structure


src/mug/            

├── cli/                    - utilities regarding an alternative code to load data and model 

├── dataset/                - dataset management (download, processing, generation )

├── nn/                     - model architecture

└── samplers/                - sample generation

└── utils/                   - utilities regarding evaluation

rois.txt                    - config file with region of interest using latitude and longitude of each city      

parameteres_variable.yaml   - parameter file with hyperparameters

script_mug_s2s.py           - main script to load the data, train, 
valid and test the sequence-to-sequence convLSTM model 

trainjupyter_s2s_main.ipynb - main notebook code to load the data, 
train, valid and test the sequence-to-sequence convLSTM model 


## Result (test images)

![](https://github.com/IBM/mug/blob/f2e701dcd4bddc056ed4287c59df6f02b79c0cec/assets/result_graphic.png)