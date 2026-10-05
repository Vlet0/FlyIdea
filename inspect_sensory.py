import pyarrow.feather as feather
df = feather.read_table('.\\data\\raw\\body-annotations-male-cns-v1.0-minconf-0.5.feather').to_pandas()
print('vnc_sensory types:')
print(df[df['superclass'] == 'vnc_sensory']['type'].value_counts().head(30))
