import pandas as pd
import re
from transformers import DataCollatorWithPadding
from datasets import Dataset
# import evaluate
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import PeftModel, PeftConfig

import torch
from sklearn.model_selection import train_test_split
import os

import torch.nn.functional as F

np.random.seed(1)

from rdkit import Chem
from rdkit.Chem import AllChem

import heapq

def join_scaffold_decorator(scaffold_smi, decorator_smi):
    scaff_mol = Chem.MolFromSmiles(scaffold_smi)
    decorator_mol = Chem.MolFromSmiles(decorator_smi)
    bond_type = Chem.BondType.SINGLE
    join_list = []

    combo = Chem.CombineMols(scaff_mol, decorator_mol)


    for atom in combo.GetAtoms():
        if atom.GetSymbol() == '*':
            join_list.append(atom.GetIdx())

    '''print('Identified *s in positions', join_list)
    print('1st * in', join_list[0])
    print('2nd * in', join_list[1])'''

    combined_scaffold = Chem.RWMol(combo)
    combined_scaffold.AddBond(join_list[0]+1, join_list[1]+1, bond_type)
    # print('After AddingBond:', Chem.MolToSmiles(combined_scaffold))
    # combined_scaffold.RemoveAtom(int(join_list[0]))
    # combined_scaffold.RemoveAtom(int(join_list[1]))

    final = Chem.MolToSmiles(combined_scaffold)
    final = final.replace('*', '').replace('()', '')

    return  final

def start_with_attachment(smiles):
    can_smiles = 'ERROR FOR ROOTING IN SMILES: ' + smiles
    # print('SMILES in start_with_attachment()', smiles)
    # m1 = Chem.MolFromSmiles(smiles[0]) ## CHECK THIS IF REQD
    m1 = Chem.MolFromSmiles(smiles)
    n = m1.GetNumAtoms()
    for i in range(n):
        a = m1.GetAtomWithIdx(i)
        s = a.GetSymbol()
        # print(a, s, i)
        if s == '*':
            can_smiles = Chem.MolToSmiles(m1, rootedAtAtom=i)

    return can_smiles
# of decorations
labels_count = 1032

checkpoint = 'data/checkpoints/checkpoint-38466/'

# base model for peft
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=labels_count)

# Load PEFT config
peft_model_dir = "/data/aqsol_sa__peft_checkpoints/"

# use this for jak1 inference
# peft_model_dir = '/data/jak1__peft_checkpoints/'

peft_config = PeftConfig.from_pretrained(peft_model_dir)

# Load base model from the config's base model name
base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)

# Load the PEFT-wrapped model using the checkpoint directory
model = PeftModel.from_pretrained(base_model, peft_model_dir)
model.eval()
model.to("cuda:0")

# decoration and their labels
df = pd.read_csv('data/All_sample_freq_600_decoration_label.csv')
df['Decoration'] = df['Decoration'].str.replace(r'\[\*\]', '*', regex=True)

# Function to identify the R-groups presen in the Scaffold
def get_r_groups(scaffold):
    # Find all occurances of R-Groups in the form [R1], [R2], etc.
    return re.findall(r"\[R\d+\]", scaffold)

def replace_r_tags(text):
    # Regular expression pattern to match [R1], [R2], etc.
    pattern = r'\[R\d+\]'
    # Replace matches with an empty string
    replaced_text = re.sub(pattern, '[Y]', text)
    return replaced_text

def decorate_scaffold(scaffold, model):
    r_groups = get_r_groups(scaffold) # Get list of R-groups
    print(r_groups)
    scaffold = replace_r_tags(scaffold)
    print(scaffold)
    prev_smiles_list, curr_smiles_list = [scaffold], [scaffold]
    # print()
    # r1_group = []
    # r2_group = []
    # r1_prob, r2_prob = [], []
    # final_smiles_list = []
    for j in range(len(r_groups)):
        smiles_list = []
        for i, smiles in enumerate(prev_smiles_list):
            curr_scaff = prev_smiles_list[i]

            # Replace the current R-group with '*' and all other R-groups with ''
            curr_scaff  = curr_scaff.replace('[Y]', '*', 1)
            
            curr_scaff = curr_scaff.replace('[Y]', '')
            # print('prev_scaff:', prev_smiles_list[i], 'curr_scaff:', curr_scaff)

            curr_scaff = curr_scaff.replace('()', '')
            # SMILES starts with *
            curr_scaff = start_with_attachment(curr_scaff)

            # # Get the decoration from BERT model
            decorations = []
            inputs = tokenizer(curr_scaff, return_tensors="pt")
            inputs = inputs.to('cuda:0')
            model = model.to('cuda:0')
            with torch.no_grad():
                logits = model(**inputs).logits
            k = 10
            top_k_decorations = torch.topk(logits, k).indices.cpu().numpy().ravel().tolist()
            # Compute probabilities using softmax
            probabilities = F.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probabilities, k=k)
            # topk_probs = values_list = tensor.cpu().numpy().flatten().tolist()
            for dec in top_k_decorations:
                decoration = df[df['label'] == dec].values.ravel().tolist()[0]
                decorations.append(decoration)
            topk_probs = topk_probs.cpu().numpy().ravel().tolist()
            print(decorations)
            print(topk_probs)
            # Combine the decoration back into the scaffold

            prev_smiles_list[i] = prev_smiles_list[i].replace('[Y]', '*', 1)
            for dec in decorations:
                # print(prev_smiles_list[i], dec)
                # dec = dec.replace('*#', '')
                try:
                    prev_smiles_list[i] = start_with_attachment(prev_smiles_list[i])
                    prev_smiles_list[i] = prev_smiles_list[i].replace('()', '')
                    smiles_list.append(join_scaffold_decorator(prev_smiles_list[i], dec))
                except:
                    print(prev_smiles_list[i], dec)
            # print('smiles_list:', smiles_list)
            print()
        prev_smiles_list = smiles_list
        # if j == 2:
        #     break
    print(prev_smiles_list)
    # return r1_group, r1_prob, r2_group, r2_prob, prev_smiles_list
scaffold_list = [
    '[R1]N1C(=O)CCC(C1=O)N1C(=O)c2c(C1=O)cccc2'
]
for scaffold in scaffold_list:
    try:
        print('Decorating the scaffold', scaffold)
        print('*'*100)
        decorate_scaffold(scaffold, model)
        print('*'*100)
    except Exception as e:
        print(f'Error: {e}')


