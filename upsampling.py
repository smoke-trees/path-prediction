import pandas as pd


all_truck = pd.read_csv('wassup_with_ti.csv')
print(all_truck.shape)
all_truck['time'] = pd.to_datetime(all_truck['time'], format = "%Y-%m-%d %H:%M:%S.%f")
all_truck = all_truck.set_index('time')

good_paths = [30,41,37962,27]
good_track_points =[]

for i in good_paths:
  good_track_points.append(all_truck.loc[all_truck['track_id']==i])

n = pd.concat(good_track_points)
seconds_resampled_data = n.resample('1S').interpolate()

l = []

for i in good_paths:
  l.append(seconds_resampled_data.loc[seconds_resampled_data['track_id']==i])

new_l = pd.concat(l)

new_l.to_csv('upscaled_new_dat_with_dir.csv')
