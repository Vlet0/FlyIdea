import pyarrow.feather as feather
df = feather.read_table('.\\data\\raw\\body-annotations-male-cns-v1.0-minconf-0.5.feather').to_pandas()
vnc = df[df['superclass'] == 'vnc_motor']['type'].value_counts()
cb = df[df['superclass'] == 'cb_motor']['type'].value_counts()
print('vnc_motor types:')
print(vnc.head(20))
print('\ncb_motor types:')
print(cb.head(20))
