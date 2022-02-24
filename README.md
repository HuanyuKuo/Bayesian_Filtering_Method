# Bayesian Filtering Method
**A Python program that applies Bayesian Filtering Method (BFM) to estimate the selection-coefficient from barcode-count data for barcode-lineage-tracking (BLT) experiment.**

The central concept of this method is the Bayesian probability distribution on the lineage's selection-coefficient, which is updated one data point at a time (prior -> posterior). At each time step, global parameters (experimental noise & population’s mean selection-coefficient) are estimated based on the current knowledge of individual lineages. All individual lineages are classified into adaptive or neutral group. This Bayesian program reportes the (1)  global parameters' trajectory and (2) estimated selection coefficients, as the mean of posterior distribution at the time with least uncertanty, for all lineages in the adaptive group. 

* Before you run the code, check **System Request** and **Data Setting**. 
* Execute the program 

  ```sh
  python3 main.py 
  ```
* You can find example of input barcode-data in "./input/Data_BarcodeCount_simuMEE_20220213.txt", with running results under "./output/"
* A simulation program is provided as a data generator for barcode-lineage-tracking. You could play with the program and use simulated data to test the BFM method code. 
* This project aims to (1) identify adpative barcode (2) estimate the selection coefficient for each adaptive lineaege (3) infer the trajectory of mean-fitness (i.e. population mean of selection coefficient) (4) estimate the systematic noise in BLT experiment.

## System Request
1. Python version >= 3.6
2. Test Libraries (Run a test code)
   ```sh
   python3 test_library.py
   ```
    my print out is: Python version 3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]
    Library version
    numpy 1.20.1
    scipy 1.6.2
    matplotlib 3.3.4
    json 2.0.9
    pickle 4.0
    noisyopt 0.2.2
    pystan 2.19.1.1
   
3. Run all .py program under "/main_scripts/model_code/". Make sure all pkl files are generated.
4. This program can use multi processors to speed up MCMC calculations. A multi CPU-core environment is highly recommended. 

## Data Setting
1. Parameters (Open "./main_scripts/myConstant.py". Edit **NUMBER_OF_PROCESSES** and **EXPERIMENTAL PARAMETERS** for your case.)
2. Input data (Save your barcode-count data as a txt file under "./input/", with barcodes=row and time-point=column)
3. Test (Edit **datafilename** in "./main_scripts/myReadfile.py" as your input file name. Then run code to test the file reading.)
    ```sh
    python3 myReadfile.py
    ```
## Run Bayesian Filtering Method
1. Open "./main_scripts/main.py". Edit **datafilename** & **case_name** for your case.
2. Run BFM in a command window. 
    ```sh
    python3 main.py
    ```
3. (Optional) Run "./main_scripts/plot_result.py" to output results or make plots.

## Debug
1. Most libraries are common in python except of "noisyopt" and "pystan". Both of them are on PyPI. I use Anacaonda (conda) to install libraries on Windows OS. For Linux user, you might use the follow command to install noisyopt and pystan.  
   ```sh
   pip3 install noisyopt
   python3 -m pip install pystan
2. If you get error message about pickle protocol (model_load = pickle.load(pickle_file) ValueError: unsupported pickle protocol: 5), it's because your current pickle version is different to the pickle version used for packing the model code. To solve this bug, re-run all model codes to generate new pkl files. 

### Folder Description
  **./main_scripts/**: python program of Barcode Filtering Method, 
  **./main_scripts/model_code/**: statistical model underline the Bayesian inferrence
  **./input/**: folder for barcode count data from BLT experiment
  **./ouput/**: folder for outputfile from this python program
  **./simulation_MEE/**: simulation program to generate barcode-count-data. 

### Contact
Huan-Yu Kuo: [linkedin,](https://www.linkedin.com/in/huan-yu-kuo/)  hukuo@ucsd.edu 

Project Link: [https://github.com/HuanyuKuo/Bayesian_Filtering_Method](https://github.com/HuanyuKuo/Bayesian_Filtering_Method)

