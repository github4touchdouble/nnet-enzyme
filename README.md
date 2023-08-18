# pbl_binary_classifier
Zu Beginn:
# HINWEIS: Environmental variables setzen
`touch .env`

# pbl_binary_classifier/.env
```
FASTA_NON_ENZYMES='PATH/TO/NON_ENZYME/FASTA'
FASTA_ENZYMES='PATH/TO/ENZYME/FASTA'

OHE_NON_ENZYMES='PATH/TO/NON_ENZYME/OHE'
OHE_ENZYMES='PATH/TO/ENZYME/OHE'

CSV30_ENZYMES='PATH/TO/ENZYME/SPLIT30'
```

#### Usage

```
import os
from dotenv import load_dotenv

load_dotenv() # load environment variables

abs_path_to_split30 = os.getenv("CSV30_ENZYMES")
abs_path_to_non_enzyme_fasta = os.getenv("FASTA_NON_ENZYMES")
```
