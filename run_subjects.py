import os, sys, datetime, pickle
import subprocess, logging, time
import pp
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import pandas as pd
from IPython import embed as shell

sys.path.append( os.environ['ANALYSIS_HOME'] )
import Tools.Operators.HDFEyeOperator
from Tools.Subjects.Subject import Subject


project_directory = '/home/degee/research/2013_pupil_yesno/data/'
raw_data = '/home/raw_data/Donner_lab/2013/visual_detection_pupil/raw/'
do_preps = False

sjs_all = [
    # # experiment 1:
    # [Subject('ab', '?', None, None, None), ['ab1_0_2012-06-19_18.23.18.edf', 'ab2_0_2012-06-21_14.26.45.edf', 'ab3_0_2012-06-22_18.00.03.edf', 'ab4_0_2012-06-22_19.00.24.edf', 'ab5_0_2012-06-25_17.43.26.edf', 'ab6_0_2012-06-25_18.39.45.edf', 'ab7_0_2012-06-28_12.20.23.edf', 'ab8_0_2012-06-28_13.16.34.edf', 'ab9_0_2012-06-29_11.50.54.edf', 'ab10_0_2012-06-29_12.56.42.edf', 'ab11_0_2012-07-02_17.44.38.edf', 'ab12_0_2012-07-02_18.42.20.edf',] ],
    # [Subject('dh', '?', None, None, None), ['dh1_0_2012-07-03_11.16.36.edf', 'dh2_0_2012-07-03_14.14.29.edf', 'dh3_0_2012-07-05_10.09.37.edf', 'dh4_0_2012-07-05_11.14.44.edf', 'dh5_0_2012-07-05_13.05.12.edf', 'dh6_0_2012-07-05_14.09.31.edf', 'dh7_0_2012-07-06_10.03.45.edf', 'dh8_0_2012-07-06_10.57.35.edf', 'dh9_0_2012-07-06_12.39.37.edf', 'dh10_0_2012-07-06_13.39.25.edf', 'dh11_0_2012-07-12_14.13.56.edf', 'dh12_0_2012-07-12_15.13.09.edf', 'dh13_0_2012-07-12_16.09.28.edf', 'dh14_0_2012-07-13_10.13.37.edf', 'dh15_0_2012-07-13_11.04.57.edf',] ],
    # [Subject('dl', '?', None, None, None), ['dl1_0_2012-09-19_13.32.13.edf', 'dl2_0_2012-09-20_15.42.58.edf', 'dl3_0_2012-09-20_16.42.36.edf', 'dl4_0_2012-09-24_11.29.41.edf', 'dl5_0_2012-09-24_12.12.51.edf', 'dl6_0_2012-09-24_14.21.11.edf', 'dl7_0_2012-09-24_15.01.23.edf', 'dl8_0_2012-09-26_12.54.37.edf', 'dl9_0_2012-09-26_13.48.36.edf', 'dl10_0_2012-09-27_16.47.16.edf', 'dl11_0_2012-09-27_17.49.19.edf', 'dl12_0_2012-10-02_15.54.52.edf', 'dl13_0_2012-10-02_16.53.26.edf', 'dl14_0_2012-10-03_13.36.02.edf', 'dl15_0_2012-10-04_16.07.07.edf', 'dl16_0_2012-10-04_16.55.29.edf', 'dl17_0_2012-10-04_17.46.30.edf',] ],
    # [Subject('rn', '?', None, None, None), ['rn1_0_2012-06-04_14.48.55.edf', 'rn2_0_2012-06-04_15.27.19.edf', 'rn3_0_2012-06-04_16.54.22.edf', 'rn4_0_2012-06-05_10.52.30.edf', 'rn5_0_2012-06-05_11.49.04.edf', 'rn6_0_2012-06-05_12.45.30.edf', 'rn7_0_2012-06-05_14.03.38.edf', 'rn8_0_2012-06-11_13.39.11.edf', 'rn9_0_2012-06-11_14.29.28.edf', 'rn10_0_2012-06-13_14.44.57.edf', 'rn11_0_2012-06-19_12.32.12.edf', 'rn12_0_2012-06-19_13.32.04.edf', 'rn13_0_2012-06-25_12.42.50.edf', 'rn14_0_2012-07-02_15.06.07.edf', 'rn15_0_2012-07-02_16.08.19.edf', 'rn16_0_2012-07-04_11.37.45.edf',] ],
    # [Subject('jl', '?', None, None, None), ['jl1_0_2012-07-03_16.07.50.edf', 'jl2_0_2012-07-04_09.51.23.edf', 'jl3_0_2012-07-05_15.36.32.edf', 'jl4_0_2012-07-05_16.24.56.edf', 'jl5_0_2012-07-05_17.10.49.edf', 'jl6_0_2012-07-10_15.19.21.edf', 'jl7_0_2012-07-10_15.40.30.edf', 'jl8_0_2012-07-10_16.12.33.edf', 'jl9_0_2012-07-10_16.33.45.edf', 'jl10_0_2012-07-10_17.03.17.edf', 'jl11_0_2012-07-10_17.23.29.edf', 'jl12_0_2012-07-11_15.58.44.edf', 'jl13_0_2012-07-11_16.37.27.edf', 'jl14_0_2012-07-11_17.13.30.edf', 'jl15_0_2012-07-11_17.31.06.edf', 'jl16_0_2012-07-11_17.58.43.edf', 'jl17_0_2012-07-12_12.18.04.edf', 'jl18_0_2012-07-12_13.08.48.edf', 'jl19_0_2012-07-13_13.13.44.edf', 'jl20_0_2012-07-13_14.12.09.edf',] ],
    # [Subject('jwg', '?', None, None, None), ['jwg1_0_2012-06-01_13.03.38.edf', 'jwg2_0_2012-06-01_13.59.05.edf', 'jwg3_0_2012-06-04_18.14.20.edf', 'jwg4_0_2012-06-05_14.58.52.edf', 'jwg5_0_2012-06-07_13.28.58.edf', 'jwg6_0_2012-06-11_16.03.32.edf', 'jwg7_0_2012-06-13_15.44.58.edf', 'jwg8_0_2012-06-13_17.19.30.edf', 'jwg9_0_2012-06-15_12.29.03.edf', 'jwg10_0_2012-06-20_13.59.52.edf', 'jwg11_0_2012-06-20_15.46.56.edf', 'jwg12_0_2012-06-21_10.57.45.edf', 'jwg13_0_2012-06-21_12.06.57.edf', 'jwg14_0_2012-06-22_11.07.49.edf', 'jwg15_0_2012-06-22_12.30.26.edf', 'jwg16_0_2012-06-26_11.50.20.edf', 'jwg17_0_2012-06-29_13.40.40.edf',] ], # caution: fix msg of run 5
    # # experiment 2:
    # [Subject('al', '?', None, None, None), ['al_1_2013-02-21_15.10.04.edf', 'al_2_2013-02-21_15.25.33.edf', 'al_3_2013-02-22_14.01.38.edf', 'al_4_2013-02-22_14.16.07.edf', 'al_5_2013-02-22_14.29.48.edf', 'al_6_2013-02-22_14.45.24.edf', 'al_7_2013-02-22_15.00.56.edf', 'al_8_2013-02-22_15.16.07.edf',], [1,1]],
    # [Subject('ch', '?', None, None, None), ['ch_1_2013-01-24_16.11.56.edf', 'ch_2_2013-01-24_16.29.08.edf', 'ch_3_2013-02-04_15.19.40.edf', 'ch_4_2013-02-04_15.35.13.edf', 'ch_5_2013-02-04_15.48.38.edf', 'ch_6_2013-02-04_16.05.07.edf', 'ch_7_2013-02-04_16.23.58.edf',]],
    # [Subject('dho', '?', None, None, None), ['dho_1_2013-02-28_16.12.43.edf', 'dho_2_2013-03-07_16.10.46.edf', 'dho_3_2013-03-07_16.25.45.edf', 'dho_4_2013-03-07_16.38.46.edf', 'dho_5_2013-03-07_16.53.10.edf', 'dho_6_2013-03-07_17.12.43.edf', 'dho_7_2013-03-07_17.25.45.edf',]],
    # [Subject('dli', '?', None, None, None), ['dli_1_2013-01-22_15.27.43.edf', 'dli_2_2013-01-23_14.11.42.edf', 'dli_3_2013-01-23_14.30.03.edf', 'dli_4_2013-01-23_14.45.51.edf', 'dli_5_2013-01-23_15.15.57.edf', 'dli_6_2013-01-23_15.31.38.edf', 'dli_7_2013-01-23_15.46.13.edf',]],
    # [Subject('fg', '?', None, None, None), ['fg_1_2013-03-21_16.10.06.edf', 'fg_2_2013-03-22_12.16.45.edf', 'fg_3_2013-03-22_12.36.01.edf', 'fg_4_2013-03-22_13.11.34.edf', 'fg_5_2013-03-25_13.43.48.edf', 'fg_6_2013-03-25_14.01.10.edf', 'fg_7_2013-03-25_14.16.32.edf', 'fg_8_2013-03-25_14.34.09.edf', 'fg_9_2013-03-22_12.53.26.edf', 'fg_10_2013-03-25_14.49.55.edf', ]],
    # [Subject('js', '?', None, None, None), ['js_1_2013-02-26_12.28.20.edf', 'js_2_2013-02-26_12.46.16.edf', 'js_3_2013-02-27_11.13.01.edf', 'js_4_2013-02-27_11.27.01.edf', 'js_5_2013-02-27_11.40.30.edf', 'js_6_2013-02-27_12.08.44.edf', 'js_7_2013-02-27_12.21.50.edf', 'js_8_2013-02-27_12.34.32.edf',]],
    # [Subject('jw', '?', None, None, None), ['jw_1_2013-01-14_14.05.09.edf', 'jw_2_2013-01-14_14.55.22.edf', 'jw_3_2013-01-15_12.32.01.edf', 'jw_4_2013-01-15_13.06.27.edf', 'jw_5_2013-01-15_13.40.02.edf',]],
    # [Subject('kr', '?', None, None, None), ['kr_1_2013-02-05_17.23.45.edf', 'kr_2_2013-02-05_17.42.13.edf', 'kr_3_2013-02-07_14.25.29.edf', 'kr_4_2013-02-07_14.43.45.edf', 'kr_5_2013-02-07_14.58.07.edf', 'kr_6_2013-02-07_15.16.56.edf', 'kr_7_2013-02-07_15.47.33.edf',]],
    # [Subject('lm', '?', None, None, None), ['lm_1_2013-01-21_14.19.57.edf', 'lm_2_2013-01-21_14.35.08.edf', 'lm_3_2013-02-06_13.19.47.edf', 'lm_4_2013-02-06_13.33.41.edf', 'lm_5_2013-02-06_13.46.47.edf', 'lm_6_2013-02-06_14.02.45.edf', 'lm_7_2013-02-06_14.23.46.edf',]],
    # [Subject('lms', '?', None, None, None), ['lms_1_2013-04-08_13.22.35.edf', 'lms_2_2013-04-08_13.38.36.edf', 'lms_3_2013-04-19_11.10.09.edf', 'lms_4_2013-04-19_11.27.01.edf', 'lms_5_2013-04-19_11.39.19.edf', 'lms_6_2013-04-19_11.58.46.edf', 'lms_7_2013-04-19_12.11.50.edf', 'lms_8_2013-04-19_12.24.52.edf',]],
    # [Subject('ln', '?', None, None, None), ['ln_1_2013-01-28_13.12.27.edf', 'ln_2_2013-01-28_13.37.41.edf', 'ln_3_2013-02-01_13.09.31.edf', 'ln_4_2013-02-01_13.26.01.edf', 'ln_5_2013-02-01_13.39.25.edf', 'ln_6_2013-02-01_14.04.14.edf', 'ln_7_2013-02-01_14.19.47.edf', 'ln_8_2013-02-01_14.32.17.edf',]],
    # [Subject('mc', '?', None, None, None), ['mc_1_2013-04-17_14.40.35.edf', 'mc_2_2013-04-18_13.50.09.edf', 'mc_3_2013-04-18_14.09.55.edf', 'mc_4_2013-04-18_14.26.32.edf', 'mc_5_2013-04-18_14.40.30.edf', 'mc_6_2013-04-18_14.53.54.edf',]],
    # [Subject('ml', '?', None, None, None), ['ml_1_2013-01-29_10.14.48.edf', 'ml_2_2013-01-29_10.35.46.edf', 'ml_3_2013-01-30_11.08.46.edf', 'ml_4_2013-01-30_11.26.05.edf', 'ml_5_2013-01-30_11.43.47.edf', 'ml_6_2013-01-30_12.17.21.edf', 'ml_7_2013-01-30_12.32.49.edf', 'ml_8_2013-01-30_12.46.57.edf',]],
    # [Subject('nk', '?', None, None, None), ['nk_1_2013-03-22_14.25.41.edf', 'nk_2_2013-04-08_14.18.20.edf', 'nk_3_2013-04-08_14.33.58.edf', 'nk_4_2013-04-08_14.52.36.edf', 'nk_5_2013-04-08_15.08.45.edf', 'nk_6_2013-04-08_15.34.57.edf',]],
    # [Subject('qr', '?', None, None, None), ['qr_1_2013-03-21_15.26.52.edf', 'qr_2_2013-03-21_15.51.17.edf', 'qr_3_2013-03-21_16.32.09.edf', 'qr_4_2013-03-22_15.16.57.edf', 'qr_5_2013-03-22_15.34.39.edf', 'qr_6_2013-03-27_13.47.00.edf', 'qr_7_2013-03-27_14.08.09.edf', 'qr_8_2013-03-27_14.26.11.edf',]],
    # [Subject('so', '?', None, None, None), ['so_1_2013-04-18_16.02.06.edf', 'so_2_2013-04-18_16.19.10.edf', 'so_3_2013-04-18_16.32.43.edf', 'so_4_2013-04-18_16.44.06.edf', 'so_5_2013-05-06_16.06.52.edf', 'so_6_2013-05-06_16.18.48.edf', 'so_7_2013-05-06_16.30.45.edf',]],
    # [Subject('td', '?', None, None, None), ['td_1_2013-01-29_19.41.45.edf', 'td_2_2013-01-29_19.57.05.edf', 'td_3_2013-01-29_20.11.02.edf', 'td_4_2013-01-29_20.25.16.edf', 'td_5_2013-02-01_16.54.27.edf', 'td_6_2013-02-01_17.25.31.edf', 'td_7_2013-02-01_17.39.45.edf', 'td_8_2013-02-01_17.58.26.edf', 'td_9_2013-02-22_18.04.17.edf', 'td_10_2013-02-22_18.16.27.edf',]],
    # [Subject('te', '?', None, None, None), ['te_1_2013-01-15_09.12.49.edf', 'te_2_2013-01-15_09.40.24.edf', 'te_3_2013-01-15_10.11.16.edf', 'te_4_2013-01-17_14.06.07.edf', 'te_5_2013-01-17_14.22.51.edf', 'te_6_2013-01-17_15.01.04.edf', 'te_7_2013-01-17_15.29.46.edf',]],
    # [Subject('tk', '?', None, None, None), ['tk_1_2013-03-10_12.22.53.edf', 'tk_2_2013-03-10_12.40.58.edf', 'tk_3_2013-03-10_13.01.03.edf', 'tk_4_2013-03-10_13.18.11.edf', 'tk_5_2013-03-10_13.53.05.edf',]],
    # [Subject('vp', '?', None, None, None), ['vp_1_2013-02-25_14.18.36.edf', 'vp_2_2013-02-25_14.33.40.edf', 'vp_3_2013-02-25_14.46.18.edf', 'vp_4_2013-02-26_13.15.53.edf', 'vp_5_2013-02-26_13.28.57.edf', 'vp_6_2013-02-26_13.42.11.edf', 'vp_7_2013-02-26_14.11.29.edf', 'vp_8_2013-02-26_14.28.46.edf',]],
    # [Subject('wb', '?', None, None, None), ['wb_1_2013-04-11_12.01.04.edf', 'wb_2_2013-04-11_12.15.23.edf', 'wb_3_2013-04-11_12.31.07.edf', 'wb_4_2013-04-15_15.36.02.edf', 'wb_5_2013-04-15_15.55.09.edf', 'wb_6_2013-04-15_16.10.04.edf', 'wb_7_2013-04-15_16.30.37.edf', 'wb_8_2013-04-15_16.42.58.edf',]],
    # [Subject('sp', '?', None, None, None), ['sp_1_2013-04-15_14.39.33.edf', 'sp_2_2013-04-15_14.57.07.edf', 'sp_3_2013-04-19_13.16.52.edf', 'sp_4_2013-04-19_13.34.45.edf', 'sp_5_2013-04-19_13.52.03.edf', 'sp_6_2013-04-19_14.11.31.edf', 'sp_7_2013-04-19_14.26.34.edf', 'sp_8_2013-04-19_14.40.40.edf',]],
    [Subject('ek', '?', None, None, None), ['ek_1_2013-02-25_12.12.38.edf', 'ek_2_2013-02-25_12.28.51.edf', 'ek_3_2013-02-27_13.07.33.edf', 'ek_4_2013-02-27_13.21.51.edf', 'ek_5_2013-02-27_13.35.41.edf', 'ek_6_2013-02-27_14.09.45.edf', 'ek_7_2013-02-27_14.23.02.edf', 'ek_8_2013-02-27_14.37.22.edf',]],
    #
    ]

def run_subject(sj, raw_data, project_directory, exp_name = 'detection_pupil'):
    
    import PupilYesNoDetection
    import numpy as np
    
    days = [file.split('201')[-1][2:7] for file in sj[1]]
    unique_days = np.unique(days)
    aliases = []
    for i in range(len(sj[1])):
        session = int(np.where(days[i] == unique_days)[0])
        aliases.append('detection_{}_{}'.format(i+1, session))
    raw_data = [os.path.join(raw_data, file) for file in sj[1]] 
    
    if sj[0].initials in ('jwg', 'rn', 'dh', 'ab', 'dl', 'jl'):
        experiment = 1
    else:
        experiment = 2
    if sj[0].initials in ('jwg', 'rn', 'dh', 'ab', 'jl', 'ch', 'dli', 'jw', 'ln', 'td', 'te', 'lm', 'fg', 'tk', 'lms', 'sp', 'so'):
        version = 1
    else:
        version = 2
    
    # # ------------------
    # # PREPROCESSING:   -
    # # ------------------
    # pupilPreprocessSession = PupilYesNoDetection.pupilPreprocessSession(subject = sj[0], experiment_name = exp_name, experiment_nr = experiment, version = version, sample_rate_new=50, project_directory = project_directory,)
    # # # pupilPreprocessSession.import_raw_data(raw_data, aliases)
    # pupilPreprocessSession.delete_hdf5()
    # pupilPreprocessSession.import_all_data(aliases)
    # for alias in aliases:
    #     pupilPreprocessSession.process_runs(alias, artifact_rejection='strict', create_pupil_BOLD_regressor=False)
    #     pass
    # pupilPreprocessSession.process_across_runs(aliases)
    
    # ------------------
    # PER SUBJECT:     -
    # ------------------
    # pupilAnalysisSession = PupilYesNoDetection.pupilAnalyses(subject=sj[0], experiment_name=exp_name, experiment_nr=experiment, sample_rate_new=50, project_directory=project_directory, aliases=aliases)
    # pupilAnalysisSession.timelocked_plots()
    # pupilAnalysisSession.behavior_confidence()
    # pupilAnalysisSession.pupil_bars()
    # pupilAnalysisSession.sequential_effects()
    # pupilAnalysisSession.GLM(aliases=aliases)
    # pupilAnalysisSession.GLM_2(aliases=aliases)
    # pupilAnalysisSession.GLM_3(aliases=aliases)
    # pupilAnalysisSession.split_by_GLM(aliases=aliases, perms=50000, run=True)
    
    return True

def analyze_group(project_directory, exp_name = 'detection_pupil'):
    
    import PupilYesNoDetection
    
    # ------------------
    # ACROSS SUBJECTS: -
    # ------------------
    # subjects = ['dh', 'ab', 'dl', 'jl', 'jwg', 'rn'] # exp1
    # subjects = ['al', 'ch', 'dho', 'dli', 'ek', 'fg', 'js', 'jw', 'kr', 'lm', 'lms', 'ln', 'mc', 'ml', 'nk', 'qr', 'so', 'sp', 'td', 'te', 'tk', 'vp', 'wb'] # exp2
    # subjects = ['ab', 'dh', 'dl', 'jwg', 'rn', 'al', 'ch', 'dho', 'dli', 'ek', 'fg', 'js', 'jw', 'kr', 'lm', 'lms', 'ln', 'mc', 'ml', 'nk', 'qr', 'so', 'sp', 'td', 'te', 'tk', 'vp', 'wb'] # exp1+2 ALL
    # subjects = ['ab', 'dh', 'dl', 'jwg', 'rn', 'al', 'ch', 'ek', 'fg', 'js', 'jw', 'lm', 'lms', 'ln', 'mc', 'ml', 'nk', 'qr', 'so', 'sp', 'td', 'tk', 'wb'] # exp1+2 SELECTED
    subjects = ['ab', 'dh', 'dl', 'rn', 'al', 'ch', 'ek', 'fg', 'js', 'jw', 'lm', 'lms', 'ln', 'mc', 'ml', 'nk', 'qr', 'so', 'sp', 'td', 'wb'] # exp1+2 SELECTED
    # subjects = ['al', 'ch', 'ek', 'fg', 'js', 'jw', 'lm', 'lms', 'ln', 'mc', 'ml', 'nk', 'qr', 'so', 'sp', 'td', 'wb'] # 2 SELECTED
    
    # subjects = ['ch']
    
    pupilAnalysisSessionAcross = PupilYesNoDetection.pupilAnalysesAcross(subjects=subjects, experiment_name=exp_name, project_directory=project_directory)
    # pupilAnalysisSessionAcross.behavior_choice()
    pupilAnalysisSessionAcross.behavior_normalized(prepare=True)
    # pupilAnalysisSessionAcross.SDT_correlation(bins=5)
    # pupilAnalysisSessionAcross.behavior_variability()
    # pupilAnalysisSessionAcross.behavior_rt_kde()
    # pupilAnalysisSessionAcross.pupil_bars()
    # pupilAnalysisSessionAcross.pupil_criterion()
    # pupilAnalysisSessionAcross.pupil_prediction_error()
    # pupilAnalysisSessionAcross.rt_distributions()
    # pupilAnalysisSessionAcross.drift_diffusion()
    # pupilAnalysisSessionAcross.pupil_signal_presence()
    # pupilAnalysisSessionAcross.GLM_betas()
    # pupilAnalysisSessionAcross.behavior_rt_kde()
    # pupilAnalysisSessionAcross.grand_average_pupil_response()
    # pupilAnalysisSessionAcross.SDT_correlation_2()
    # pupilAnalysisSessionAcross.SDT_across_time()
    # pupilAnalysisSessionAcross.binned_baseline_pupil_behavior()
    # pupilAnalysisSessionAcross.mean_slow_drift()
    
    return True

def analyze_subjects(sjs_all):
    if len(sjs_all) > 1: 
        job_server = pp.Server(ppservers=())
        start_time = time.time()
        jobs = [(sj, job_server.submit(run_subject,(sj, raw_data, project_directory), (), ('PupilYesNoDetection',))) for sj in sjs_all]
        results = []
        for s, job in jobs:
            job()
        print "Time elapsed: ", time.time() - start_time, "s"
        job_server.print_stats()
    else:
        run_subject(sjs_all[0], raw_data=raw_data, project_directory=project_directory)

def main(do_preps=do_preps):
    analyze_subjects(sjs_all)
    analyze_group(project_directory=project_directory, exp_name='detection_pupil')

if __name__ == '__main__':
    main()

# 