import os, sys, datetime, pickle
import subprocess, logging, time
import pp
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import pandas as pd
from IPython import embed as shell
import glob

sys.path.append( os.environ['ANALYSIS_HOME'] )
import Tools.Operators.HDFEyeOperator
from Tools.Subjects.Subject import Subject

project_directory = '/home/degee/research/2013_pupil_yesno/data2/'
raw_data = '/home/raw_data/UvA/Donner_lab/2017_eLife/2_pupil_yesno_visual/'

sjs_all = []
subjects = [
            'sub-01',
            'sub-02',
            'sub-03',
            'sub-04',
            'sub-05',
            'sub-06',
            'sub-07',
            'sub-08',
            'sub-09',
            'sub-10',
            'sub-11',
            'sub-12',
            'sub-13',
            'sub-14',
            'sub-15',
            'sub-16',
            'sub-17',
            'sub-18',
            'sub-19',
            'sub-20',
            'sub-21',
            'sub-22',
            'sub-23',
            'sub-24',
            'sub-25',
            'sub-26',
            'sub-27',
            'sub-28',
            ]

for s in subjects:
    runs = np.sort([r.split('/')[-1] for r in glob.glob(os.path.join(raw_data, s, 'sub-*'))])
    sjs_all.append([Subject(s, '?', None, None, None), runs],)
    
def run_subject(sj, raw_data, project_directory, exp_name = 'detection_pupil'):
    
    import defs_pupil
    import numpy as np
    
    session_nrs = [int(f.split('ses-')[-1][:1]) for f in sj[1]]
    aliases = []
    for i in range(len(sj[1])):
        aliases.append('detection_{}_{}'.format(i+1, session_nrs[i]))
    raw_data = [os.path.join(raw_data, sj[0].initials, f) for f in sj[1]] 
    
    if sj[0].initials in ('sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'):
        experiment = 1
    else:
        experiment = 2
    if sj[0].initials in ['sub-01', 'sub-02', 'sub-04', 'sub-05', 'sub-07', 'sub-09', 'sub-11', 'sub-13', 'sub-15', 'sub-16', 'sub-17', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26']:
        version = 1
    else:
        version = 2
    
    # ------------------
    # PREPROCESSING:   -
    # ------------------
    # pupilPreprocessSession = defs_pupil.pupilPreprocessSession(subject = sj[0], experiment_name = exp_name, experiment_nr = experiment, version = version, sample_rate_new=50, project_directory = project_directory,)
    # pupilPreprocessSession.import_raw_data(raw_data, aliases)
    # pupilPreprocessSession.delete_hdf5()
    # pupilPreprocessSession.import_all_data(aliases)
    # for alias in aliases:
    #     pupilPreprocessSession.process_runs(alias, artifact_rejection='strict', create_pupil_BOLD_regressor=False)
    #     pass
    # pupilPreprocessSession.process_across_runs(aliases)
    
    # ------------------
    # PER SUBJECT:     -
    # ------------------
    # pupilAnalysisSession = defs_pupil.pupilAnalyses(subject=sj[0], experiment_name=exp_name, experiment_nr=experiment, sample_rate_new=50, project_directory=project_directory, aliases=aliases)
    # pupilAnalysisSession.GLM(aliases=aliases)
    
    return True

def analyze_group(project_directory, exp_name = 'detection_pupil'):
    
    import defs_pupil
    
    # ------------------
    # ACROSS SUBJECTS: -
    # ------------------
    
    # exp1:
    # subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
    # exp2:
    # subjects = ['sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28']
    # exp1+2 ALL:
    # subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28']
    # exp1+2 (N-23 --> main analysis in PNAS paper, without 5 subjects with decision-related pupil constrictions):
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-26', 'sub-28']
    # exp1+2 (N-21 --> main analysis in eLife paper, because sub-04 and sub-26 already appeared in the fMRI data set):
    # subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06', 'sub-07', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-28']
    
    # pupilAnalysisSessionAcross = defs_pupil.pupilAnalysesAcross(subjects=subjects, experiment_name=exp_name, project_directory=project_directory)
    # pupilAnalysisSessionAcross.behavior_choice()
    # pupilAnalysisSessionAcross.behavior_normalized(prepare=False)
    # pupilAnalysisSessionAcross.SDT_correlation(bins=5)
    # pupilAnalysisSessionAcross.rt_distributions()
    # pupilAnalysisSessionAcross.drift_diffusion()
    # pupilAnalysisSessionAcross.average_pupil_responses()
    # pupilAnalysisSessionAcross.grand_average_pupil_response()
    # pupilAnalysisSessionAcross.SDT_across_time()
    # pupilAnalysisSessionAcross.correlation_PPRa_BPD()
    # pupilAnalysisSessionAcross.GLM_betas()
    
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

def main():
    analyze_subjects(sjs_all)
    analyze_group(project_directory=project_directory, exp_name='detection_pupil')

if __name__ == '__main__':
    main()