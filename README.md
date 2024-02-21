# *nnet-enzyme*: A Framework designed for the Classification of Enzymatic Protein Sequences
> **Note:** For further insights into the *nnet-enzyme* framework, refer to the corresponding paper accessible [here](https://github.com/github4touchdouble/nnet-enzyme/blob/malte/Report/main.pdf).
> 
Malte A. Weyrich, Than Truc Bui, Jan P. Hummel, Sofiia Kozoriz 

Technical University of Munich | Technische Universität München

Contact via: [jan.hummel@tum.de](mailto:jan.hummel@tum.de)

## Getting started

Follow these steps to set up and configure the environment for your specific needs:

1. **Environment Setup:**
   - Clone this repository `git clone https://github.com/github4touchdouble/nnet-enzyme.git`
   - Install the necessary dependencies and libraries. Ensure compatibility with your system specifications `pip install -r './requirements.txt'`
   - Create an .env file in the root of the project  `nano .env` and add the following lines and adjust for your individual needs:
     
      ```python
      # --------------------------------
      #    Non-enzymatic protein data 
      # --------------------------------

      # <WIP>
      FASTA_NON_ENZYMES='PATH/TO/NON_ENZYME/FASTA'
      FASTA_ENZYMES='PATH/TO/ENZYME/FASTA'
      PROTT5_NON_ENZYMES='PATH/TO/NON_ENZYME/PROTT5' -- optional: absed on your needs
      ESM2_NON_ENZYMES='PATH/TO/NON_ENZYME/ESM2' # -- optional: absed on your needs
      OHE_NON_ENZYMES='PATH/TO/NON_ENZYME/OHE' # >> i.e. provide one-hot-encoded protein sequences

      # ----------------------------
      #    Enzymatic protein data 
      # ----------------------------

      # Enzyme, enzyme commission number and amino acid sequence
      # CSV file: <Identifier>,<EC>,<Sequence> ~> C7C422,3.5.2.6,MEL...KLR
      # I.a. if you intend to train models using datasets with varying levels of redundancy reduction, replace "X" with the required percentage of similarity for two sequences to be deemed duplicates
      # Customize this as needed for your specific requirements. Refer to the "Run configuration" section for ESSENTIAL considerations before intiating a project
      CSVX_ENZYMES='PATH/TO/ENZYME/SPLITX'

      # Enzyme, protein embedding vector
      # H5 file: <Identifier>,<Embedding> ~> A0A024RBG1,[-0.015143169, 0.035552002, -0.02231326, ...]
      # I.a. if you intend to train models using datasets with varying levels of redundancy reduction, replace "X" with the required percentage of similarity for two sequences to be deemed duplicates
      # I.a. nnet-enzyme offers support for ESM2, PROTT5, and One-hot encoded vectors
      # Customize this as needed for your specific requirements. Refer to the "Run configuration" section for ESSENTIAL considerations before intiating a project
      ESM2_ENZYMES_SPLIT_X='PATH/TO/ENZYME/ESM2/SPLIT_X' # i.a.
      PROTT5_ENZYMES_SPLIT_X='PATH/TO/ENZYME/PROTT5/SPLIT_X' # i.a.     
      OHE_ENZYMES_SPLIT_X='PATH/TO/ENZYME/OHE' # i.a.
      ```

2. **Run configuration:**
   > **WIP**
4. **Classification Pipelines:**
   - Execute the provided Jupyter notebooks to follow the classification pipeline as detailed in the accompanying paper.
   - Should you require the integration of *nnet-enzyme* into a custom pipeline, adapt the provided code to align with the requirements of your framework. Merge relevant components seamlessly to ensure smooth functionality within your project.

## Example of use 
> **WIP**

## Supplementary information: Unstructured tips & tricks 

#### Using the environment variables in the code
```python
import os
from dotenv import load_dotenv

load_dotenv() # load environment variables, should return True

abs_path_to_split30 = os.getenv("CSV30_ENZYMES")
abs_path_to_non_enzyme_fasta = os.getenv("FASTA_NON_ENZYMES")

[...]
```

#### Adding .env to .gitignore

Make sure to add the `.env` file to the `.gitignore` file so that the environment variables are not pushed to the repository.

In `.gitignore` add the following line:
```
.env
```

#### Reading embeddings

First import the H5Dataset class:

```python
form data_manipulation import load_ml_data
```
Then use this method to load the data:
#### Loading enzyme esm2 embeddings
```python
enzyme_csv = os.getenv("CSVX_ENZYMES") # replace X with the number of the split you want to use
enzyme_esm2 = os.getenv("ESM2_ENZYMES_SPLIT_X") # replace X with the number of the split you want to use

X_enzymes, y_enzymes = load_ml_data(path_to_esm2=enzyme_esm2, path_tp_enzyme_csv=enzyme_csv)
```
#### Loading non enzyme esm2 embeddings
Since we don't have a `.csv` for our non enzymes, we need to use the `load_non_enzyme_esm2` method instead:
```python
path_to_non_ez_fasta = os.getenv("FASTA_NON_ENZYMES")
path_to_non_ez_esm2 = os.getenv("ESM2_NON_ENZYMES")
X_non_enzymes, y_non_enzymes  = load_non_enzyme_esm2(non_enzymes_fasta_path = path_to_non_ez_fasta, non_enzymes_esm2_path=path_to_non_ez_esm2)
```
#### Now we can merge the two datasets:**
```python
# Combine data
X = np.vstack((X_enzymes, X_non_enzymes))
y = np.hstack((y_enzymes, y_non_enzymes))
```
