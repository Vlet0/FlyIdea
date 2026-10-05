import pyarrow.feather as feather
import pandas as pd
df = feather.read_table('.\\data\\raw\\body-annotations-male-cns-v1.0-minconf-0.5.feather').to_pandas()
print('Total neurons:', len(df))
print('Unique superclasses:')
print(df['superclass'].value_counts(dropna=False))
print('\nTypes with leg/wing/haltere/neck in name:')
print(df[df['type'].str.contains('(?i)wing|leg|haltere|neck', na=False)]['type'].unique())
