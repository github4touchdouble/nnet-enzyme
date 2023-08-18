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
```

### Using the environment variables in the code

```python
import os
from dotenv import load_dotenv

load_dotenv() # load environment variables

abs_path_to_split30 = os.getenv("CSV30_ENZYMES")
abs_path_to_non_enzyme_fasta = os.getenv("FASTA_NON_ENZYMES")
```
