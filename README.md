# Bayesian Filtering Method
A Python program that applies Bayesian filtering method to estimate the selection-coefficient from barcode-lineage-tracking data.

The central concept of this method is the Bayesian probability distribution on the lineage's selection-coefficient, which is updated one data point at a time (prior -> posterior). At each time step, global parameters (experimental noise & population’s mean selection-coefficient) are estimated based on the current knowledge of individual lineages. All individual lineage is classified into adaptive or neutral class. The Bayesian estimate of selection coefficient, for single lineage, is simply the mean of posterior distribution of individual adpative lineage. 

* Before you run the code, check System Request and Data Setting. 
* Execute the program 
  ```sh
  python ./main_scripts/main.py
  ```
## System Request
1. Python version > 2.6
2. Libraries (Run a test code to check libraries)
   ```sh
   python ./main_scripts/test_library.py
   ```
    my print out is: Python version 3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]
    Library version
    numpy 1.20.1
    scipy 1.6.2
    matplotlib 3.3.4
    json 2.0.9
    pickle 4.0
    noisyopt 0.2.2
   
3. This program can use multi-processing to speed up the MCMC calculations. A multi CPU-core environment is recommended. 
## Data Setting
1. Parameters (Open "./main_scripts/myConstant.py". Edit **NUMBER_OF_PROCESSES** and **EXPERIMENTAL PARAMETERS** for your case.)
2. Input data (Save your barcode-count data as a txt file under "./input/", with barcodes=row and time-point=column)
3. Test (Edit **datafilename** in "./main_scripts/myReadfile.py" as above input file name. Then run code to test the file reading.)
    ```sh
    python ./main_scripts/myReadfile.py
    ```
## Run Bayesian Filtering Method
1. Open "./main_scripts/main.py". Edit **datafilename** & **case_name** for your case.
2. Run "./main_scripts/main.py" in a command window. 
    ```sh
    python ./main_scripts/main.py
    ```
3. (Optional) Run "./main_scripts/plot_result.py" to output results or make plots.

### Folder Description
  ./main_scripts/: python program of Barcode Filtering Method

  ./input/: folder for barcode count data from lineage tracking experiment

  ./ouput/: folder for outputfile from this python program

  ./main_scripts/model_code/: statistical model underline the Bayesian inferrence

  ./simulation_MEE/: simulation program to generate barcode-count-data. A simulation program is provided as data generator for barcode-lineage-tracking. You could play with it and use simulation data to test the BFM code. 

### Contact
Huan-Yu Kuo: [linkedin,](https://www.linkedin.com/in/huan-yu-kuo/)  -hukuo@ucsd.edu 

Project Link: [https://github.com/HuanyuKuo/Bayesian_Filtering_Method](https://github.com/HuanyuKuo/Bayesian_Filtering_Method)
