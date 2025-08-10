import numpy as np
import pandas as pd
import statsmodels.api as sm

# == Import data from csv ==

# PROSIT
chunks_prosit = pd.read_csv("raw_data/export_prosit.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'env_illuminance', 'env_illuminance_d', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate']]
chunks_prosit['subject_id'] = chunks_prosit['session_description'].str.split('-').str.get(0)
sessions_prosit = chunks_prosit.drop_duplicates(subset='session_id')
subjects_prosit = sessions_prosit.drop_duplicates(subset='subject_id')
# VV
chunks_vv = pd.read_csv("raw_data/export_vv_medium.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'env_illuminance', 'env_illuminance_d', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate']]
chunks_vv['subject_id'] = chunks_vv['session_description'].str.split('/').str.get(1)
sessions_vv = chunks_vv.drop_duplicates(subset='session_id')
# VV-Africa
chunks_vv_africa = pd.read_csv("raw_data/export_vv_africa_small.csv")[['id', 'camera_stationary', 'duration', 'env_complexity', 'env_variability', 'env_illuminance', 'env_illuminance_d', 'subject_age', 'subject_facemask', 'subject_gaze', 'subject_gender', 'subject_glasses', 'subject_headwear', 'subject_illuminance_d', 'subject_movement', 'subject_skin_type', 'frame_avg_bp_dia', 'frame_avg_bp_sys', 'frame_avg_hr_pox', 'frame_avg_hrv_sdnn_ecg', 'frame_avg_rr', 'frame_avg_spo2', 'session_description', 'session_id', 'session_split', 'session_video_framerate']]
chunks_vv_africa['subject_id'] = chunks_vv_africa['session_description'].str.split('/').str.get(1)
sessions_vv_africa = chunks_vv_africa.drop_duplicates(subset='session_id')

# == Dataset summaries ==

# PROSIT
splits = ['train', 'valid', 'test']
prosit_summary = pd.DataFrame({
  'split': [s.capitalize() for s in splits],
  'participants': [chunks_prosit[chunks_prosit['session_split'] == s]['subject_id'].nunique() for s in splits],
  'sessions': [chunks_prosit[chunks_prosit['session_split'] == s]['session_description'].nunique() for s in splits],
  'chunks': [chunks_prosit[chunks_prosit['session_split'] == s].shape[0] for s in splits],
  'time': [round(chunks_prosit[chunks_prosit['session_split'] == s]['duration'].sum()/(60.*60.), 1) for s in splits],
})
prosit_summary.to_csv("data/prosit_summary.csv", index=False)
# VV-Medium
vv_medium_summary = pd.DataFrame({
  'split': ['Test'],
  'participants': [int(chunks_vv['subject_id'].nunique())],
  'chunks': [chunks_vv.shape[0]],
  'time': [round(chunks_vv['duration'].sum()/(60.*60.), 1)],
})
vv_medium_summary.to_csv("data/vv_medium_summary.csv", index=False)
# VV-Africa-Small
vv_africa_small_summary = pd.DataFrame({
  'split': [s.capitalize() for s in splits],
  'participants': [chunks_vv_africa[chunks_vv_africa['session_split'] == s]['subject_id'].nunique() for s in splits],
  'chunks': [chunks_vv_africa[chunks_vv_africa['session_split'] == s].shape[0] for s in splits],
  'time': [round(chunks_vv_africa[chunks_vv_africa['session_split'] == s]['duration'].sum()/(60.*60.), 1) for s in splits],
})
vv_africa_small_summary.to_csv("data/vv_africa_small_summary.csv", index=False)
# Training data)
sources = ['PROSIT', 'VV-Africa-Small']
training_summary = pd.DataFrame({
  'source': sources,
  'participants': [prosit_summary[prosit_summary['split'] == 'Train']['participants'].values[0] if s == 'PROSIT' else vv_africa_small_summary[vv_africa_small_summary['split'] == 'Train']['participants'].values[0] for s in sources],
  'chunks': [prosit_summary[prosit_summary['split'] == 'Train']['chunks'].values[0] if s == 'PROSIT' else vv_africa_small_summary[vv_africa_small_summary['split'] == 'Train']['chunks'].values[0] for s in sources],
  'time': [prosit_summary[prosit_summary['split'] == 'Train']['time'].values[0] if s == 'PROSIT' else vv_africa_small_summary[vv_africa_small_summary['split'] == 'Train']['time'].values[0] for s in sources],
})
training_summary.to_csv("data/training_summary.csv", index=False)

# == Demographics ==

# Training Dataset Age
frequencies, bin_edges = np.histogram(np.concatenate([subjects_prosit[subjects_prosit['session_split'] == 'train']['subject_age'],
                                                      sessions_vv_africa[sessions_vv_africa['session_split'] == 'train']['subject_age']], axis=-1),
                                      bins=8)
np.savetxt('data/age_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# Training Dataset Gender
counts_prosit = subjects_prosit[subjects_prosit['session_split'] == 'train']['subject_gender'].value_counts()
counts_vv_africa = sessions_vv_africa[sessions_vv_africa['session_split'] == 'train']['subject_gender'].value_counts()
counts = counts_prosit.add(counts_vv_africa, fill_value=0).astype(int)
counts_df = pd.DataFrame({'Label': counts.index.str.capitalize(), 'Value': counts.values})
counts_df.to_csv("data/gender.csv", index=False)

# Training Dataset Skin Type
counts_prosit = subjects_prosit[subjects_prosit['session_split'] == 'train']['subject_skin_type'].value_counts()
counts_vv_africa = sessions_vv_africa[sessions_vv_africa['session_split'] == 'train']['subject_skin_type'].value_counts()
counts = counts_prosit.add(counts_vv_africa, fill_value=0).astype(int)
counts_df = pd.DataFrame({'Label': counts.index.map(lambda val: "Type {}".format(val)), 'Value': counts.values})
counts_df = counts_df.sort_values(by='Label')
counts_df.to_csv("data/skin_type.csv", index=False)

# == Summary vitals ==

# Training Dataset HR
frequencies, bin_edges = np.histogram(np.concatenate([chunks_prosit[chunks_prosit['session_split'] == 'train']['frame_avg_hr_pox'],
                                                      chunks_vv_africa[chunks_vv_africa['session_split'] == 'train']['frame_avg_hr_pox']], axis=-1),
                                      bins=10)
np.savetxt('data/hr_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# Training Dataset RR
frequencies, bin_edges = np.histogram(np.concatenate([chunks_prosit[(chunks_prosit['session_split'] == 'train') & (chunks_prosit['frame_avg_rr'] >= 0) & (chunks_prosit['frame_avg_rr'] <= 40)]['frame_avg_rr'],
                                                      chunks_vv_africa[chunks_vv_africa['session_split'] == 'train']['frame_avg_rr']], axis=-1),
                                      bins=10)
np.savetxt('data/rr_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# Training Dataset BP
frequencies_sys, bin_edges_sys = np.histogram(np.concatenate([chunks_prosit[chunks_prosit['session_split'] == 'train']['frame_avg_bp_sys'],
                                                              chunks_vv_africa[chunks_vv_africa['session_split'] == 'train']['frame_avg_bp_sys']], axis=-1),
                                              bins=10)
frequencies_dia, bin_edges_dia = np.histogram(np.concatenate([chunks_prosit[chunks_prosit['session_split'] == 'train']['frame_avg_bp_dia'],
                                                              chunks_vv_africa[chunks_vv_africa['session_split'] == 'train']['frame_avg_bp_dia']], axis=-1),
                                              bins=10)
np.savetxt('data/bp_sys_histogram.csv', np.column_stack(((bin_edges_sys[:-1]+bin_edges_sys[1:])/2, frequencies_sys)), delimiter=',', header='BinEdges,Frequency', comments='')
np.savetxt('data/bp_dia_histogram.csv', np.column_stack(((bin_edges_dia[:-1]+bin_edges_dia[1:])/2, frequencies_dia)), delimiter=',', header='BinEdges,Frequency', comments='')

# Training Dataset SpO2
frequencies, bin_edges = np.histogram(np.concatenate([chunks_prosit[(chunks_prosit['session_split'] == 'train') & (chunks_prosit['frame_avg_spo2'] >= 50)]['frame_avg_spo2'],
                                                      chunks_vv_africa[(chunks_vv_africa['session_split'] == 'train') & (chunks_vv_africa['frame_avg_spo2'] >= 50)]['frame_avg_spo2']], axis=-1),
                                      bins=10)
np.savetxt('data/spo2_histogram.csv', np.column_stack(((bin_edges[:-1]+bin_edges[1:])/2, frequencies)), delimiter=',', header='BinEdges,Frequency', comments='')

# == Results ==

# Import for PROSIT
def merge_force_suffix(left, right, **kwargs):
  on_col = kwargs['on']
  suffix_tupple = kwargs['suffixes']
  def suffix_col(col, suffix):
    if col != on_col:
      return str(col) + suffix
    else:
      return col
  left_suffixed = left.rename(columns=lambda x: suffix_col(x, suffix_tupple[0]))
  right_suffixed = right.rename(columns=lambda x: suffix_col(x, suffix_tupple[1]))
  del kwargs['suffixes']
  return pd.merge(left_suffixed, right_suffixed, **kwargs)
raw_plethnet_prosit = pd.read_csv("raw_data/evaluate_plethnet_v2_72_prosit_test_30.csv")
raw_plethnet_prosit = raw_plethnet_prosit[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_plethnet_prosit = raw_plethnet_prosit.rename(columns={'chunk_id': 'id'})
results_prosit_test = merge_force_suffix(chunks_prosit, raw_plethnet_prosit, on='id', how='inner', suffixes=['', '_vl'])
raw_g_prosit = pd.read_csv("raw_data/evaluate_g_prosit.csv")
raw_g_prosit = raw_g_prosit[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_g_prosit = raw_g_prosit.rename(columns={'chunk_id': 'id'})
results_prosit_test = merge_force_suffix(results_prosit_test, raw_g_prosit, on='id', how='inner', suffixes=['', '_g'])
raw_chrom_prosit = pd.read_csv("raw_data/evaluate_chrom_prosit.csv")
raw_chrom_prosit = raw_chrom_prosit[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_chrom_prosit = raw_chrom_prosit.rename(columns={'chunk_id': 'id'})
results_prosit_test = merge_force_suffix(results_prosit_test, raw_chrom_prosit, on='id', how='inner', suffixes=['', '_chrom'])
raw_pos_prosit = pd.read_csv("raw_data/evaluate_pos_prosit.csv")
raw_pos_prosit = raw_pos_prosit[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_pos_prosit = raw_pos_prosit.rename(columns={'chunk_id': 'id'})
results_prosit_test = merge_force_suffix(results_prosit_test, raw_pos_prosit, on='id', how='inner', suffixes=['', '_pos'])
raw_deep_phys_prosit = pd.read_csv("raw_data/evaluate_deep_phys_prosit.csv")
raw_deep_phys_prosit = raw_deep_phys_prosit[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_deep_phys_prosit = raw_deep_phys_prosit.rename(columns={'chunk_id': 'id'})
results_prosit_test = merge_force_suffix(results_prosit_test, raw_deep_phys_prosit, on='id', how='inner', suffixes=['', '_deepphys'])
raw_mtts_can_prosit = pd.read_csv("raw_data/evaluate_mtts_can_prosit_8.csv")
raw_mtts_can_prosit = raw_mtts_can_prosit[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_mtts_can_prosit = raw_mtts_can_prosit.rename(columns={'chunk_id': 'id'})
results_prosit_test = merge_force_suffix(results_prosit_test, raw_mtts_can_prosit, on='id', how='inner', suffixes=['', '_mtts'])

# Import for VV
raw_plethnet_vv = pd.read_csv("raw_data/evaluate_plethnet_v2_72_vv_30.csv")
raw_plethnet_vv = raw_plethnet_vv[['chunk_id', 'gt_hr_pox', 'gt_rr', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_plethnet_vv = raw_plethnet_vv.rename(columns={'chunk_id': 'id'})
results_vv = merge_force_suffix(pd.concat([chunks_vv, chunks_vv_africa], ignore_index=True), raw_plethnet_vv, on='id', how='inner', suffixes=['', '_vl'])
raw_g_vv = pd.read_csv("raw_data/evaluate_g_vv.csv")
raw_g_vv = raw_g_vv[['chunk_id', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_g_vv = raw_g_vv.rename(columns={'chunk_id': 'id'})
results_vv = merge_force_suffix(results_vv, raw_g_vv, on='id', how='inner', suffixes=['', '_g'])
raw_chrom_vv = pd.read_csv("raw_data/evaluate_chrom_vv.csv")
raw_chrom_vv = raw_chrom_vv[['chunk_id', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_chrom_vv = raw_chrom_vv.rename(columns={'chunk_id': 'id'})
results_vv = merge_force_suffix(results_vv, raw_chrom_vv, on='id', how='inner', suffixes=['', '_chrom'])
raw_pos_vv = pd.read_csv("raw_data/evaluate_pos_vv.csv")
raw_pos_vv = raw_pos_vv[['chunk_id', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_pos_vv = raw_pos_vv.rename(columns={'chunk_id': 'id'})
results_vv = merge_force_suffix(results_vv, raw_pos_vv, on='id', how='inner', suffixes=['', '_pos'])
raw_deep_phys_vv = pd.read_csv("raw_data/evaluate_deep_phys_vv.csv")
raw_deep_phys_vv = raw_deep_phys_vv[['chunk_id', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_deep_phys_vv = raw_deep_phys_vv.rename(columns={'chunk_id': 'id'})
results_vv = merge_force_suffix(results_vv, raw_deep_phys_vv, on='id', how='inner', suffixes=['', '_deepphys'])
raw_mtts_can_vv = pd.read_csv("raw_data/evaluate_mtts_can_vv_8.csv")
raw_mtts_can_vv = raw_mtts_can_vv[['chunk_id', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_mtts_can_vv = raw_mtts_can_vv.rename(columns={'chunk_id': 'id'})
results_vv = merge_force_suffix(results_vv, raw_mtts_can_vv, on='id', how='inner', suffixes=['', '_mtts'])
raw_plethnet_64_vv = pd.read_csv("raw_data/evaluate_plethnet_v2_64_vv.csv")
raw_plethnet_64_vv = raw_plethnet_64_vv[['chunk_id', 'pred_hr', 'pred_rr', 'conf_pulse_mean', 'conf_resp_mean', 'pulse_mae', 'pulse_mse', 'pulse_rmse', 'pulse_cor', 'pulse_snr', 'hr_ae', 'resp_mae', 'resp_mse', 'resp_rmse', 'resp_cor', 'resp_snr', 'rr_ae', 'live_error', 'live_error_t']]
raw_plethnet_64_vv = raw_plethnet_64_vv.rename(columns={'chunk_id': 'id'})
results_vv = merge_force_suffix(results_vv, raw_plethnet_64_vv, on='id', how='inner', suffixes=['', '_vl_64'])

methods = ['g', 'chrom', 'pos', 'deepphys', 'mtts', 'vl']
methods_print = ['G', 'CHROM', 'POS', 'DeepPhys', 'MTTS-CAN', 'VitalLens']

## VV-Medium
results_vv_medium = results_vv[~results_vv['session_description'].str.contains('ghana')]
results_vv_medium_summary = pd.DataFrame({
  'method': methods_print,
  'hr_mae': [results_vv_medium['hr_ae_{}'.format(m)].mean() for m in methods],
  'pulse_snr': [results_vv_medium['pulse_snr_{}'.format(m)].mean() for m in methods],
  'pulse_cor': [results_vv_medium['pulse_cor_{}'.format(m)].mean() for m in methods],
  'rr_mae': [results_vv_medium['rr_ae_{}'.format(m)].mean() for m in methods],
  'resp_snr': [results_vv_medium['resp_snr_{}'.format(m)].mean() for m in methods],
  'resp_cor': [results_vv_medium['resp_cor_{}'.format(m)].mean() for m in methods],
  'inf_speed': [3.4, 4.2, 3.6, 9.8, 22.1, 18.0]
}).fillna(0)
results_vv_medium_summary.to_csv("data/results_vv_medium.csv", index=False)

## VV-Africa Test Set
results_vv_africa_test = results_vv[(results_vv['session_description'].str.contains('ghana')) & (results_vv['session_split'] == 'test')]
results_vv_africa_test_summary = pd.DataFrame({
  'method': methods_print,
  'hr_mae': [results_vv_africa_test['hr_ae_{}'.format(m)].mean() for m in methods],
  'pulse_snr': [results_vv_africa_test['pulse_snr_{}'.format(m)].mean() for m in methods],
  'pulse_cor': [results_vv_africa_test['pulse_cor_{}'.format(m)].mean() for m in methods],
  'rr_mae': [results_vv_africa_test['rr_ae_{}'.format(m)].mean() for m in methods],
  'resp_snr': [results_vv_africa_test['resp_snr_{}'.format(m)].mean() for m in methods],
  'resp_cor': [results_vv_africa_test['resp_cor_{}'.format(m)].mean() for m in methods],
}).fillna(0)
results_vv_africa_test_summary.to_csv("data/results_vv_africa_test.csv", index=False)

## PROSIT Test Set
results_prosit_test_summary = pd.DataFrame({
  'method': methods_print,
  'hr_mae': [results_prosit_test['hr_ae_{}'.format(m)].mean() for m in methods],
  'pulse_snr': [results_prosit_test['pulse_snr_{}'.format(m)].mean() for m in methods],
  'pulse_cor': [results_prosit_test['pulse_cor_{}'.format(m)].mean() for m in methods],
  'rr_mae': [results_prosit_test['rr_ae_{}'.format(m)].mean() for m in methods],
  'resp_snr': [results_prosit_test['resp_snr_{}'.format(m)].mean() for m in methods],
  'resp_cor': [results_prosit_test['resp_cor_{}'.format(m)].mean() for m in methods],
}).fillna(0)
results_prosit_test_summary.to_csv("data/results_prosit_test.csv", index=False)

# == Regression ==

## VV-Medium - HR SNR
categorical_cols = ['subject_gender', 'subject_skin_type']
numerical_cols = ['subject_age', 'subject_illuminance_d', 'subject_movement', 'frame_avg_hr_pox']
vv_medium_pulse_snr_reg = results_vv_medium[categorical_cols + numerical_cols + ['pulse_snr_vl']].dropna()
X_categorical = pd.get_dummies(vv_medium_pulse_snr_reg[categorical_cols], columns=categorical_cols, drop_first=True)
X = pd.concat([vv_medium_pulse_snr_reg[numerical_cols], X_categorical], axis=1).astype('float64')
X = sm.add_constant(X)
y = vv_medium_pulse_snr_reg['pulse_snr_vl'].astype(float)
model = sm.OLS(y, X).fit()
print("VV-Medium HR")
print(model.summary())
print(model.summary().as_latex())

## PROSIT - HR SNR
categorical_cols = ['camera_stationary']
numerical_cols = ['subject_illuminance_d', 'subject_movement']
prosit_pulse_snr_reg = results_prosit_test[categorical_cols + numerical_cols + ['pulse_snr_vl']].dropna()
X_categorical = pd.get_dummies(prosit_pulse_snr_reg[categorical_cols], columns=categorical_cols, drop_first=True)
X = pd.concat([prosit_pulse_snr_reg[numerical_cols], X_categorical], axis=1).astype('float64')
X = sm.add_constant(X)
y = prosit_pulse_snr_reg['pulse_snr_vl'].astype(float)
model = sm.OLS(y, X).fit()
print("PROSIT HR")
print(model.summary())
print(model.summary().as_latex())

## VV-Medium - RR SNR
categorical_cols = ['subject_gender', 'subject_skin_type']
numerical_cols = ['subject_age', 'subject_illuminance_d', 'subject_movement', 'frame_avg_rr']
vv_medium_resp_snr_reg = results_vv_medium[categorical_cols + numerical_cols + ['resp_snr_vl']].dropna()
X_categorical = pd.get_dummies(vv_medium_resp_snr_reg[categorical_cols], columns=categorical_cols, drop_first=True)
X = pd.concat([vv_medium_resp_snr_reg[numerical_cols], X_categorical], axis=1).astype('float64')
X = sm.add_constant(X)
y = vv_medium_resp_snr_reg['resp_snr_vl'].astype(float)
model = sm.OLS(y, X).fit()
print("VV-Medium RR")
print(model.summary())
print(model.summary().as_latex())

## PROSIT - RR SNR
categorical_cols = ['camera_stationary']
numerical_cols = ['subject_illuminance_d', 'subject_movement']
prosit_resp_snr_reg = results_prosit_test[categorical_cols + numerical_cols + ['resp_snr_vl']].dropna()
X_categorical = pd.get_dummies(prosit_resp_snr_reg[categorical_cols], columns=categorical_cols, drop_first=True)
X = pd.concat([prosit_resp_snr_reg[numerical_cols], X_categorical], axis=1).astype('float64')
X = sm.add_constant(X)
y = prosit_resp_snr_reg['resp_snr_vl'].astype(float)
model = sm.OLS(y, X).fit()
print("PROSIT RR")
print(model.summary())
print(model.summary().as_latex())

# == Bar charts ==

## subject_movement
bins = [0., .05, .1, .15, .25, 1.]
bin_names = ["No movements", "Few movements", "Some movements", "Moderate movement", "Extreme movement"]
impact_movement = pd.DataFrame({
  'bin_left': bins[:-1],
  'bin_right': bins[1:],
  'bin': ["{} [{:.2f}; {:.2f}]".format(n, l, r) for n, l, r in zip(bin_names, bins[:-1], bins[1:])],
  'n': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_movement'], bins))['pulse_snr_vl'].count().reset_index()['pulse_snr_vl'],
  'pulse_snr_mean': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_movement'], bins))['pulse_snr_vl'].mean().reset_index()['pulse_snr_vl'],
  'pulse_snr_sd': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_movement'], bins))['pulse_snr_vl'].std().reset_index()['pulse_snr_vl'],
  'resp_snr_mean': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_movement'], bins))['resp_snr_vl'].mean().reset_index()['resp_snr_vl'],
  'resp_snr_sd': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_movement'], bins))['resp_snr_vl'].std().reset_index()['resp_snr_vl'],
}).fillna(0)
impact_movement.to_csv("data/impact_movement.csv", index=False)

## subject_illuminance_d
bins = [0., .05, .1, .15, .25, 1.]
bin_names = ["Perfect illuminance", "Few variation", "Some variation", "Moderate variation", "Extreme variation"]
impact_illuminance = pd.DataFrame({
  'bin_left': bins[:-1],
  'bin_right': bins[1:],
  'bin': ["{} [{:.2f}; {:.2f}]".format(n, l, r) for n, l, r in zip(bin_names, bins[:-1], bins[1:])],
  'n': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_illuminance_d'], bins))['pulse_snr_vl'].count().reset_index()['pulse_snr_vl'],
  'pulse_snr_mean': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_illuminance_d'], bins))['pulse_snr_vl'].mean().reset_index()['pulse_snr_vl'],
  'pulse_snr_sd': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_illuminance_d'], bins))['pulse_snr_vl'].std().reset_index()['pulse_snr_vl'],
  'resp_snr_mean': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_illuminance_d'], bins))['resp_snr_vl'].mean().reset_index()['resp_snr_vl'],
  'resp_snr_sd': results_prosit_test.groupby(pd.cut(results_prosit_test['subject_illuminance_d'], bins))['resp_snr_vl'].std().reset_index()['resp_snr_vl']
}).fillna(0)
impact_illuminance.to_csv("data/impact_illuminance.csv", index=False)

## subject_age
bins = [0, 20, 40, 60, 80, 100]
impact_age = pd.DataFrame({
  'bin': ["[{}; {}]".format(l, r) for l, r in zip(bins[:-1], bins[1:])],
  'n': results_vv_medium.groupby(pd.cut(results_vv_medium['subject_age'], bins))['pulse_snr_vl'].count().reset_index()['pulse_snr_vl'],
  'pulse_snr_mean': results_vv_medium.groupby(pd.cut(results_vv_medium['subject_age'], bins))['pulse_snr_vl'].mean().reset_index()['pulse_snr_vl'],
  'pulse_snr_sd': results_vv_medium.groupby(pd.cut(results_vv_medium['subject_age'], bins))['pulse_snr_vl'].std().reset_index()['pulse_snr_vl'],
  'resp_snr_mean': results_vv_medium.groupby(pd.cut(results_vv_medium['subject_age'], bins))['resp_snr_vl'].mean().reset_index()['resp_snr_vl'],
  'resp_snr_sd': results_vv_medium.groupby(pd.cut(results_vv_medium['subject_age'], bins))['resp_snr_vl'].std().reset_index()['resp_snr_vl']
}).fillna(0)
impact_age.to_csv("data/impact_age.csv", index=False)

## skin_type
impact_skin_type = pd.DataFrame({
  'bin': [1, 2, 3, 4, 5, 6],
  'n': results_vv_medium.groupby('subject_skin_type')['pulse_snr_vl'].count().reset_index()['pulse_snr_vl'],
  'pulse_snr_mean': results_vv_medium.groupby('subject_skin_type')['pulse_snr_vl'].mean().reset_index()['pulse_snr_vl'],
  'pulse_snr_sd': results_vv_medium.groupby('subject_skin_type')['pulse_snr_vl'].std().reset_index()['pulse_snr_vl'],
  'resp_snr_mean': results_vv_medium.groupby('subject_skin_type')['resp_snr_vl'].mean().reset_index()['resp_snr_vl'],
  'resp_snr_sd': results_vv_medium.groupby('subject_skin_type')['resp_snr_vl'].std().reset_index()['resp_snr_vl']
}).fillna(0)
impact_skin_type.to_csv("data/impact_skin_type.csv", index=False)
impact_skin_type_comparison = pd.DataFrame({
  'bin': [1, 2, 3, 4, 5, 6],
  'n': results_vv_medium.groupby('subject_skin_type')['pulse_snr_vl'].count().reset_index()['pulse_snr_vl'],
  'pulse_snr_mean_vl': results_vv_medium.groupby('subject_skin_type')['pulse_snr_vl'].mean().reset_index()['pulse_snr_vl'],
  'pulse_snr_sd_vl': results_vv_medium.groupby('subject_skin_type')['pulse_snr_vl'].std().reset_index()['pulse_snr_vl'],
  'pulse_snr_mean_pos': results_vv_medium.groupby('subject_skin_type')['pulse_snr_pos'].mean().reset_index()['pulse_snr_pos'],
  'pulse_snr_sd_pos': results_vv_medium.groupby('subject_skin_type')['pulse_snr_pos'].std().reset_index()['pulse_snr_pos'],
  'pulse_snr_mean_vl_64': results_vv_medium.groupby('subject_skin_type')['pulse_snr_vl_64'].mean().reset_index()['pulse_snr_vl_64'],
  'pulse_snr_sd_vl_64': results_vv_medium.groupby('subject_skin_type')['pulse_snr_vl_64'].std().reset_index()['pulse_snr_vl_64'],
}).fillna(0)
impact_skin_type_comparison.to_csv("data/impact_skin_type_comparison.csv", index=False)

## hr
bins = [0., 50., 60., 70., 80., 90., 100., 110., 130.]
impact_hr = pd.DataFrame({
  'bin_left': bins[:-1],
  'bin_right': bins[1:],
  'bin': ["[{:.0f}; {:.0f}]".format(l, r) for l, r in zip(bins[:-1], bins[1:])],
  'n': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_hr_pox'], bins))['pulse_snr_vl'].count().reset_index()['pulse_snr_vl'],
  'pulse_snr_mean': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_hr_pox'], bins))['pulse_snr_vl'].mean().reset_index()['pulse_snr_vl'],
  'pulse_snr_sd': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_hr_pox'], bins))['pulse_snr_vl'].std().reset_index()['pulse_snr_vl'],
  'resp_snr_mean': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_hr_pox'], bins))['resp_snr_vl'].mean().reset_index()['resp_snr_vl'],
  'resp_snr_sd': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_hr_pox'], bins))['resp_snr_vl'].std().reset_index()['resp_snr_vl']
}).fillna(0)
impact_hr.to_csv("data/impact_hr.csv", index=False)

## rr
bins = [0., 10., 15., 20., 25., 40.]
impact_rr = pd.DataFrame({
  'bin_left': bins[:-1],
  'bin_right': bins[1:],
  'bin': ["[{:.0f}; {:.0f}]".format(l, r) for l, r in zip(bins[:-1], bins[1:])],
  'n': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_rr'], bins))['pulse_snr_vl'].count().reset_index()['pulse_snr_vl'],
  'pulse_snr_mean': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_rr'], bins))['pulse_snr_vl'].mean().reset_index()['pulse_snr_vl'],
  'pulse_snr_sd': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_rr'], bins))['pulse_snr_vl'].std().reset_index()['pulse_snr_vl'],
  'resp_snr_mean': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_rr'], bins))['resp_snr_vl'].mean().reset_index()['resp_snr_vl'],
  'resp_snr_sd': results_vv_medium.groupby(pd.cut(results_vv_medium['frame_avg_rr'], bins))['resp_snr_vl'].std().reset_index()['resp_snr_vl']
}).fillna(0)
impact_rr.to_csv("data/impact_rr.csv", index=False)


correlation_matrix = chunks_prosit[['frame_avg_hr_pox', 'subject_movement', 'subject_illuminance_d', 'subject_age']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# == Scatterplots ==

## hr VV-Medium
results_vv_medium = results_vv[~results_vv['session_description'].str.contains('ghana')]
results_vv_medium_scatter = pd.DataFrame({
  'hr_est': results_vv_medium['pred_hr_vl'],
  'hr_true': results_vv_medium['gt_hr_pox_vl'],
  'rr_est': results_vv_medium['pred_rr_vl'],
  'rr_true': results_vv_medium['gt_rr_vl'],
}).fillna(0)
results_vv_medium_scatter.to_csv("data/results_vv_medium_scatter.csv", index=False)

