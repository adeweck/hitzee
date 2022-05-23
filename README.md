# hitzee
Normalization of plate-based molecular hit screening.

To install first run:
```
pip install git+https://github.com/adeweck/hitzee.git
```

To run:
```
python

import hitzee

input_dir = 'path/to/assay/values/'
annot_dir = 'path/to/assay/annotations/'
output_file = 'hit_zscores.csv'

hit_df = hitzee(input_dir, annot_dir, output_file)

hit_df.head()
```

input_dir should contain csv files

annot_dir should contain txt files

