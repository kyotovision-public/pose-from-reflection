import numpy as np
import os
import matplotlib.pyplot as plt

result_dir = './run/eval_test_inv_gbr/test_inv_gbr'
out_dir = './run/plot_inv_gbr_results'
fontsize = 22
fontsize_legend=15
figwidth = 16
figheight_3cols = 10
fontname="Times New Roman"
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(1,1, figsize = (0.5 * figwidth, 0.38 * figheight_3cols), squeeze = False, tight_layout = True)
fig_rate, ax_rate = plt.subplots(3,1, figsize = (figwidth, figheight_3cols), squeeze = False, tight_layout = True)
fig_perr, ax_perr = plt.subplots(3,1, figsize = (figwidth, figheight_3cols), squeeze = False, tight_layout = True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = fontsize_legend

list_num_corrs_nm = []
#list_worst_compound_error = []
list_failure_rate_compound = []
for num_corrs_nm in np.arange(15):
    num_corrs_rm = 0
    result_file = result_dir+'/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)+'.npz'

    if not os.path.exists(result_file):
        continue

    result = np.load(result_file)
    list_num_corrs_nm.append(num_corrs_nm)
    list_failure_rate_compound.append(result['failure_rate_compound_001'])
ax[0][0].plot(list_num_corrs_nm, list_failure_rate_compound, '--x', label='1st Step (Estimation of $G_{21}$)')

for num_corrs_nm in [1,2,3,4,5,10,20,100]:
    list_num_corrs_rm = []
    list_mean_pose_error_deg = []
    list_median_pose_error_deg = []
    list_worst_pose_error_deg = []
    list_failure_rate_01deg = []
    list_failure_rate_1deg = []
    list_failure_rate_10deg = []
    for num_corrs_rm in np.arange(21):
        result_file = result_dir+'/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)+'.npz'

        if not os.path.exists(result_file):
            continue

        result = np.load(result_file)
        list_num_corrs_rm.append(num_corrs_rm)
        list_mean_pose_error_deg.append(result['mean_pose_error_deg'])
        list_median_pose_error_deg.append(result['median_pose_error_deg'])
        list_worst_pose_error_deg.append(result['worst_pose_error_deg'])
        list_failure_rate_01deg.append(result['failure_rate_0.1deg'])
        list_failure_rate_1deg.append(result['failure_rate_1deg'])
        list_failure_rate_10deg.append(result['failure_rate_10deg'])

    ax_perr[0][0].semilogy(list_num_corrs_rm, list_mean_pose_error_deg, '--x', label='# of NM Correspondences:'+str(num_corrs_nm))
    ax_perr[1][0].plot(list_num_corrs_rm, list_median_pose_error_deg, '--x', label='# of NM Correspondences:'+str(num_corrs_nm))
    ax_perr[2][0].semilogy(list_num_corrs_rm, list_worst_pose_error_deg, '--x', label='# of NM Correspondences:'+str(num_corrs_nm))


    ax_rate[0][0].semilogy(list_num_corrs_rm, list_failure_rate_01deg, '--x', label='# of NM Correspondences:'+str(num_corrs_nm))
    ax_rate[1][0].semilogy(list_num_corrs_rm, list_failure_rate_1deg, '--x', label='# of NM Correspondences:'+str(num_corrs_nm))
    ax_rate[2][0].semilogy(list_num_corrs_rm, list_failure_rate_10deg, '--x', label='# of NM Correspondences:'+str(num_corrs_nm))

    if num_corrs_nm == 4:
        ax[0][0].plot(list_num_corrs_rm[:15], list_failure_rate_1deg[:15], '--x', label='2nd Step (Decomposition of $G_{21}$)')
    #ax[1][0].plot(list_num_corrs_rm[:15], list_mean_pose_error_deg[:15], '--x', label='# of NM Correspondences:'+str(num_corrs_nm))



xticks = np.arange(21)
for _ in range(3):
    ax_rate[_][0].grid()
    ax_rate[_][0].set_xticks(xticks)
    ax_rate[_][0].legend(loc='upper right')

    ax_perr[_][0].grid()
    ax_perr[_][0].set_xticks(xticks)
    ax_perr[_][0].legend(loc='upper right')

for _ in range(len(ax)):
    ax[_][0].grid()
    ax[_][0].set_xticks(xticks[:15])
    ax[_][0].legend(loc='upper right')
    ax[_][0].tick_params(labelsize=0.8*fontsize)

ax_rate[2][0].set_xlabel('# of RM Correspondences', fontsize=fontsize, fontname=fontname)
ax_rate[0][0].set_ylabel('Failure Rate (>0.1deg)', fontsize=fontsize, fontname=fontname)
ax_rate[1][0].set_ylabel('Failure Rate (>1deg)', fontsize=fontsize, fontname=fontname)
ax_rate[2][0].set_ylabel('Failure Rate (>10deg)', fontsize=fontsize, fontname=fontname)

ax_perr[2][0].set_xlabel('# of RM Correspondences', fontsize=fontsize, fontname=fontname)
ax_perr[0][0].set_ylabel('Mean Error [deg]', fontsize=fontsize, fontname=fontname)
ax_perr[1][0].set_ylabel('Median Error [deg]', fontsize=fontsize, fontname=fontname)
ax_perr[2][0].set_ylabel('Worst Error [deg]', fontsize=fontsize, fontname=fontname)

ax[-1][0].set_xlabel('# of Correspondences', fontsize=fontsize, fontname=fontname)
ax[0][0].set_ylabel('Failure Rate', fontsize=fontsize, fontname=fontname)
#ax[1][0].set_ylabel('Mean Error [deg]', fontsize=fontsize, fontname=fontname)

fig_rate.savefig(out_dir+'/nrm_vs_failure_rate.pdf')
fig_perr.savefig(out_dir+'/nrm_vs_pose_error.pdf')
fig.savefig(out_dir+'/nrm_vs_rate_and_error.pdf')

# The number of NM correspondences vs Results
fig_rate, ax_rate = plt.subplots(3,1, figsize = (figwidth, figheight_3cols), squeeze = False, tight_layout = True)
fig_perr, ax_perr = plt.subplots(3,1, figsize = (figwidth, figheight_3cols), squeeze = False, tight_layout = True)


for num_corrs_rm in [1,2,3,4,5,10,20,100]:
    list_num_corrs_nm = []
    list_mean_pose_error_deg = []
    list_median_pose_error_deg = []
    list_worst_pose_error_deg = []
    list_failure_rate_01deg = []
    list_failure_rate_1deg = []
    list_failure_rate_10deg = []
    for num_corrs_nm in np.arange(21):
        result_file = result_dir+'/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)+'.npz'

        if not os.path.exists(result_file):
            continue

        result = np.load(result_file)
        list_num_corrs_nm.append(num_corrs_nm)
        list_mean_pose_error_deg.append(result['mean_pose_error_deg'])
        list_median_pose_error_deg.append(result['median_pose_error_deg'])
        list_worst_pose_error_deg.append(result['worst_pose_error_deg'])
        list_failure_rate_01deg.append(result['failure_rate_0.1deg'])
        list_failure_rate_1deg.append(result['failure_rate_1deg'])
        list_failure_rate_10deg.append(result['failure_rate_10deg'])

    ax_perr[0][0].semilogy(list_num_corrs_nm, list_mean_pose_error_deg, '--x', label='# of RM Correspondences:'+str(num_corrs_rm))
    ax_perr[1][0].plot(list_num_corrs_nm, list_median_pose_error_deg, '--x', label='# of RM Correspondences:'+str(num_corrs_rm))
    ax_perr[2][0].semilogy(list_num_corrs_nm, list_worst_pose_error_deg, '--x', label='# of RM Correspondences:'+str(num_corrs_rm))


    ax_rate[0][0].semilogy(list_num_corrs_nm, list_failure_rate_01deg, '--x', label='# of RM Correspondences:'+str(num_corrs_rm))
    ax_rate[1][0].semilogy(list_num_corrs_nm, list_failure_rate_1deg, '--x', label='# of RM Correspondences:'+str(num_corrs_rm))
    ax_rate[2][0].semilogy(list_num_corrs_nm, list_failure_rate_10deg, '--x', label='# of RM Correspondences:'+str(num_corrs_rm))



xticks = np.arange(21)
for _ in range(3):
    ax_rate[_][0].grid()
    ax_rate[_][0].set_xticks(xticks)
    ax_rate[_][0].legend(loc='upper right')

    ax_perr[_][0].grid()
    ax_perr[_][0].set_xticks(xticks)
    ax_perr[_][0].legend(loc='upper right')

ax_rate[2][0].set_xlabel('# of NM Correspondences', fontsize=fontsize, fontname=fontname)
ax_rate[0][0].set_ylabel('Failure Rate (>0.1deg)', fontsize=fontsize, fontname=fontname)
ax_rate[1][0].set_ylabel('Failure Rate (>1deg)', fontsize=fontsize, fontname=fontname)
ax_rate[2][0].set_ylabel('Failure Rate (>10deg)', fontsize=fontsize, fontname=fontname)

ax_perr[2][0].set_xlabel('# of NM Correspondences', fontsize=fontsize, fontname=fontname)
ax_perr[0][0].set_ylabel('Mean Error [deg]', fontsize=fontsize, fontname=fontname)
ax_perr[1][0].set_ylabel('Median Error [deg]', fontsize=fontsize, fontname=fontname)
ax_perr[2][0].set_ylabel('Worst Error [deg]', fontsize=fontsize, fontname=fontname)

fig_rate.savefig(out_dir+'/nnm_vs_failure_rate.pdf')
fig_perr.savefig(out_dir+'/nnm_vs_pose_error.pdf')

fig_compound, ax_compound = plt.subplots(3,1, figsize = (figwidth, figheight_3cols), squeeze = False, tight_layout = True)

list_num_corrs_nm = []
list_mean_compound_error = []
list_median_compound_error = []
#list_worst_compound_error = []
list_failure_rate_compound = []
num_corrs_rm = 0
for num_corrs_nm in np.arange(21):
    result_file = result_dir+'/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)+'.npz'

    if not os.path.exists(result_file):
        continue

    result = np.load(result_file)
    list_num_corrs_nm.append(num_corrs_nm)
    list_mean_compound_error.append(result['mean_compound_mat_error'])
    list_median_compound_error.append(result['median_compound_mat_error'])
    list_failure_rate_compound.append(result['failure_rate_compound_001'])

ax_compound[0][0].semilogy(list_num_corrs_nm, list_failure_rate_compound, '--x')
ax_compound[1][0].semilogy(list_num_corrs_nm, list_mean_compound_error, '--x')
ax_compound[2][0].plot(list_num_corrs_nm, list_median_compound_error, '--x')

xticks = np.arange(21)
for _ in range(3):
    ax_compound[_][0].grid()
    ax_compound[_][0].set_xticks(xticks)
    #ax_compound[_][0].legend(loc='upper right')

ax_compound[2][0].set_xlabel('# of NM Correspondences', fontsize=fontsize, fontname=fontname)
ax_compound[0][0].set_ylabel('Failure Rate (>0.01)', fontsize=fontsize, fontname=fontname)
ax_compound[1][0].set_ylabel('Mean Error', fontsize=fontsize, fontname=fontname)
ax_compound[2][0].set_ylabel('Median Error', fontsize=fontsize, fontname=fontname)

fig_compound.savefig(out_dir+'/nnm_vs_compound_estimation_accuracy.pdf')

plt.show()
