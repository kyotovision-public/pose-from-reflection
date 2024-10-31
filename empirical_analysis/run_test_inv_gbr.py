import subprocess
import os


for num_corrs_nm in range(6):
    proc_list = []
    for num_corrs_rm in range(6):
        if (num_corrs_nm == 0) and (num_corrs_rm == 0):
            continue
        result_dir = './run/test_inv_gbr/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)
        if os.path.exists(result_dir):
            print('Num of Corrs in NM:', num_corrs_nm)
            print('Num of Corrs in RM:', num_corrs_rm)
            print('Results already exist')
            continue
        proc = subprocess.Popen([
            'python',
            'empirical_analysis/test_inv_gbr.py',
            '-nr', str(num_corrs_rm),
            '-nn', str(num_corrs_nm),
        ])
        proc_list.append(proc)
    for subproc in proc_list:
        subproc.wait()

proc_list = []
for num_corrs_nm in [100,20,10,5,4,3,2,1]:
    for num_corrs_rm in range(15):
        if (num_corrs_nm == 0) and (num_corrs_rm == 0):
            continue
        result_dir = './run/test_inv_gbr/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)
        if os.path.exists(result_dir):
            print('Num of Corrs in NM:', num_corrs_nm)
            print('Num of Corrs in RM:', num_corrs_rm)
            print('Results already exist')
            continue
        proc = subprocess.Popen([
            'python',
            'empirical_analysis/test_inv_gbr.py',
            '-nr', str(num_corrs_rm),
            '-nn', str(num_corrs_nm),
        ])
        proc_list.append(proc)
        if len(proc_list) == 10:
            for subproc in proc_list:
                subproc.wait()
        proc_list = []

for num_corrs_nm in range(6):
    proc_list = []
    for num_corrs_pix in range(6):
        num_corrs_rm = 0
        if (num_corrs_nm == 0) and (num_corrs_rm == 0):
            continue
        result_dir = './run/test_inv_gbr/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)
        if num_corrs_nm != num_corrs_pix:
            result_dir = './run/test_inv_gbr/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)+'_'+str(num_corrs_pix)
        if os.path.exists(result_dir):
            print('Num of Corrs in NM:', num_corrs_nm)
            print('Num of Corrs in RM:', num_corrs_rm)
            print('Num of Corrs in Im:', num_corrs_pix)
            print('Results already exist')
            continue
        proc = subprocess.Popen([
            'python',
            'empirical_analysis/test_inv_gbr.py',
            '-nr', str(num_corrs_rm),
            '-nn', str(num_corrs_nm),
            '-np', str(num_corrs_pix),
        ])
        proc_list.append(proc)
    for subproc in proc_list:
        subproc.wait()
