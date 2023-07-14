# pbl_binary_classifier

# HINWEIS: Environmental variables setzen
`touch notebooks/env.py`

```
#pbl_binary_classifier/notebooks/env.py

import os

def set():
    os.environ["NON_ENZYMES"] = "PATH/TO/FASTA"
    os.environ["ENZYMES"] = 'PATH/TO/CSV'
```
Anschließend Kernel (neu) starten
