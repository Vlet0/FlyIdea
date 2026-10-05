import pyarrow.feather as feather
df = feather.read_table('.\\data\\raw\\body-annotations-male-cns-v1.0-minconf-0.5.feather').to_pandas()
cb = df[df['superclass'] == 'cb_intrinsic']['type'].value_counts()
print('Top cb_intrinsic types:')
print(cb.head(30))
