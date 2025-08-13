## MolDecor: Leveraging transformers to decorate bioactive molecules for property optimization 

## Authors: Dibyajyoti Das, Sarveswara Rao Vangala and Arijit Roy* 

## TCS Research (Life Sciences Division), Tata Consultancy Services Limited, Hyderabad 500081, India 
## Corresponding author: roy.arijit3@tcs.com 

# Implementation of "MolDecor: Leveraging transformers to decorate bioactive molecules for property optimization"

# Requirements - Preferably a conda environment with all these packages installed

* rdkit-pypi
* transformers
* evaluate
* datasets
* scikit-learn
* torch==2.1.2
* accelerate
* peft


# Order of running the code files.
#------------------------------------------------------------------------------------------
1.Pre-training the bert-base model from the huggingface path: 'unikei/bert-base-smiles'

>> $ python - u 1_lead_opt_bert.py

Note: Input -> Curated Scaffold-decoration pairs in csv with labels (only a sample dataset is given)
	  Output-> Pretrained model store in data/checkpoints directory and Class_weights.csv
#------------------------------------------------------------------------------------------
2.Fine-tuning the pretrained model on the Solubility dataset / affinity dataset

>> $ python -u 2_lead_opt_fine_tune.py

Note: Input -> Pretrained checkpoint file from step 1 and Solubility dataset
	  Output-> Fine-tuned model for Solubility decoration
#------------------------------------------------------------------------------------------
3. Getting inference from model for Solubility / Affinity, 
Note. Change the checkpoint path while running for respective target dataset

>> $ python -u 3_lead_opt_inference.py

Note: Input -> Fine-tuned checkpoint file from step 2/3 and a scaffold with attachment point (star)
	  Output-> Top -k predicted decorations for the given scaffold

# Code usage
Instructions on how to use the code have been provided in the indivual code files. For queries related to code usage, contact the corresponding authors for more information.