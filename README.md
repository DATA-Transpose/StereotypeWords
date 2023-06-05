# Words Can Be Confusing: Stereotype Bias Removal in Text Classification at the Word Level

***
Code and data for the paper **[Words Can Be Confusing: Stereotype Bias Removal in Text Classification at the Word Level](https://link.springer.com/chapter/10.1007/978-3-031-33383-5_8)**


Shen, S., Zhang, M., Chen, W., Bialkowski, A., & Xu, M.. Words Can Be Confusing: Stereotype Bias Removal in Text Classification at the Word Level. PAKDD 2023, Osaka, Japan, May 25–28, 2023, Proceedings.
***

## Requirements:

First, setup the textdebias environment and install all necessary packages:

    conda env create -f textdebias.yaml
    
## Data Introduce:  

The data folder includes the the experiment data on Amazon dataset. As for the other six dataset, you can refer to the repo [Corsair](https://github.com/qianc62/Corsair). The data is in a Json file includes dictionaries as ('label': '0', 'text':'text data').

The data in the Json file should be splited into words by space
   
## Results Reproduce:  

You can use different parameters in config.py file for different model and datasets. You can also run 

    python my_main.py --Dataset_Name DATASET --Base_Model MODEL

    
## Citation:
If you refer to the code, please cite this paper:

    @inproceedings{DBLP:conf/pakdd/ShenZCBX23,
    author       = {Shaofei Shen and 
                    Mingzhe Zhang and 
                    Weitong Chen and
                    Alina Bialkowski and 
                    Miao Xu},
    title        = {Words Can Be Confusing: Stereotype Bias Removal in Text Classification
                    at the Word Level},
    booktitle    = {{PAKDD} 2023, Osaka,
                    Japan, May 25-28, 2023, Proceedings, Part {IV}},
    year         = {2023},
  }
    
   
