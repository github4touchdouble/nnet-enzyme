# pbl_binary_classifier

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

# Meta data

A metadata .csv file is located under `metadata/lable_dict.csv`. It contains the following columns:

| Header | EC Number |
| ...    | ...       |

When matching the EC number (as label) to our embeddings, we can just use this file as a lookup table. This way we save storage space and time.
