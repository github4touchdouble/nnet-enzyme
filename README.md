# pbl_binary_classifier

## HINWEIS: Environmental variables setzen

**Create an own .env file in the root directory of the project** 

`touch .env`

**Now add the following lines to the .env file and replace the values with the absolute paths to the files on your system:**

```python
FASTA_NON_ENZYMES='PATH/TO/NON_ENZYME/FASTA'
FASTA_ENZYMES='PATH/TO/ENZYME/FASTA'

OHE_NON_ENZYMES='PATH/TO/NON_ENZYME/OHE'
OHE_ENZYMES='PATH/TO/ENZYME/OHE'

CSV30_ENZYMES='PATH/TO/ENZYME/SPLIT30'
CSV100_ENZYMES='PATH/TO/ENZYME/SPLIT100'

ESM2_NON_ENZYMES_SPLIT_X='PATH/TO/NON_ENZYME/ESM2/SPLIT_X'
ESM2_ENZYMES_SPLIT_X='PATH/TO/ENZYME/ESM2/SPLIT_X'

PROTT5_NON_ENZYMES_SPLIT_X='PATH/TO/NON_ENZYME/PROTT5/SPLIT_X'
PROTT5_ENZYMES_SPLIT_X='PATH/TO/ENZYME/PROTT5/SPLIT_X'
```

Make sure to replace the X in the last two lines with the number of the split you want to use. Also make 
sure to adapt to the naming scheme of the files on your system. The file variables need to be the same for all of us,
but the paths they point to are unique to each of us.

### Using the environment variables in the code

```python
import os
from dotenv import load_dotenv

load_dotenv() # load environment variables

abs_path_to_split30 = os.getenv("CSV30_ENZYMES")
abs_path_to_non_enzyme_fasta = os.getenv("FASTA_NON_ENZYMES")
```

### Adding .env to .gitignore

Make sure to add the .env file to the .gitignore file so that the environment variables are not pushed to the repository.

In .gitignore add the following line:

```
.env
```

