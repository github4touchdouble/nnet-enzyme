# nnet-enzyme
The corresponting paper to nnet-enzyme can be found [here](https://github.com/github4touchdouble/nnet-enzyme/blob/malte/Report/main.pdf)

## IMPORTANT: Setting up the environment

**Create an own .env file in the root directory of the project** 

`touch .env`

**Now add the following lines to the .env file and replace the values with the absolute paths to the files on your system**
Open .env and fill in the following variables with the absolute paths to the files on your system:

```python
FASTA_NON_ENZYMES='PATH/TO/NON_ENZYME/FASTA'
FASTA_ENZYMES='PATH/TO/ENZYME/FASTA'

OHE_NON_ENZYMES='PATH/TO/NON_ENZYME/OHE'
OHE_ENZYMES='PATH/TO/ENZYME/OHE'

CSV30_ENZYMES='PATH/TO/ENZYME/SPLIT30'
CSV100_ENZYMES='PATH/TO/ENZYME/SPLIT100'

ESM2_NON_ENZYMES='PATH/TO/NON_ENZYME/ESM2'
ESM2_ENZYMES_SPLIT_X='PATH/TO/ENZYME/ESM2/SPLIT_X'

PROTT5_NON_ENZYMES='PATH/TO/NON_ENZYME/PROTT5'
PROTT5_ENZYMES_SPLIT_X='PATH/TO/ENZYME/PROTT5/SPLIT_X'
```

Make sure to replace the X in the last couple of lines with the number of the split you want to use. Also make 
sure to adapt to the naming scheme of the files on your system. The variables need to be the same for all of us,
but the paths they point to are unique to each of us.

### Using the environment variables in the code
```python
import os
from dotenv import load_dotenv

load_dotenv() # load environment variables, should return True

abs_path_to_split30 = os.getenv("CSV30_ENZYMES")
abs_path_to_non_enzyme_fasta = os.getenv("FASTA_NON_ENZYMES")

[...]
```

### Adding .env to .gitignore

Make sure to add the `.env` file to the `.gitignore` file so that the environment variables are not pushed to the repository.

In `.gitignore` add the following line:
```
.env
```

# Reading embeddings with script of Tobias

First import the H5Dataset class Tobias provided:

```python
form data_manipulation import load_ml_data
```
Then use this method to load the data:
## Loading enzyme esm2 embeddings
```python
enzyme_csv = os.getenv("CSVX_ENZYMES") # replace X with the number of the split you want to use
enzyme_esm2 = os.getenv("ESM2_ENZYMES_SPLIT_X") # replace X with the number of the split you want to use

X_enzymes, y_enzymes = load_ml_data(path_to_esm2=enzyme_esm2, path_tp_enzyme_csv=enzyme_csv)
```
## Loading non enzyme esm2 embeddings
Since we don't have a `.csv` for our non enzymes, we need to use the `load_non_enzyme_esm2` method instead:
```python
path_to_non_ez_fasta = os.getenv("FASTA_NON_ENZYMES")
path_to_non_ez_esm2 = os.getenv("ESM2_NON_ENZYMES")
X_non_enzymes, y_non_enzymes  = load_non_enzyme_esm2(non_enzymes_fasta_path = path_to_non_ez_fasta, non_enzymes_esm2_path=path_to_non_ez_esm2)
```
**Now we can merge the two datasets:**
```python
# Combine data
X = np.vstack((X_enzymes, X_non_enzymes))
y = np.hstack((y_enzymes, y_non_enzymes))
```
