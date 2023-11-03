import numpy as np
import matplotlib.pyplot as plt

dataset = 'mnist'
mode = 'train'
mode = 'test'

file_path_epoch = f'./saved_loss/{dataset}/epoch_loss/'
file_path_step = f'./saved_loss/{dataset}/step_loss/'
file_name_list = ['normal', 'mid_0.0', 'mid_1e-09', 'mid_1e-06', 'mid_0.001', 'mid_1', 'mid_100', 'mid_10000']
file_name_list = ['normal', 'mid_0.0', 'mid_1e-09', 'mid_1e-06', 'mid_0.001', 'mid_1', 'mid_100']
# file_name_list = ['normal', 'mid_0.0', 'mid_1e-09', 'mid_1e-06', 'mid_0.001', 'mid_1']
# file_name_list = ['normal', 'mid_0.0', 'mid_1e-09', 'mid_1e-06']
# file_name_list = ['normal', 'mid_0.0']
epoch_loss_list = []
step_loss_list = []

for file_name in file_name_list:
    # epoch_loss = np.load(f'{file_path_epoch}{file_name}_backup.npy')
    # step_loss = np.load(f'{file_path_step}{file_name}_backup.npy')
    epoch_loss = np.load(f'{file_path_epoch}{file_name}_{mode}.npy')
    step_loss = np.load(f'{file_path_step}{file_name}_{mode}.npy')
    epoch_loss_list.append(epoch_loss)
    step_loss_list.append(step_loss)

epoch_loss_list = np.asarray(epoch_loss_list)
step_loss_list = np.asarray(step_loss_list)
epoch_loss_list = np.transpose(epoch_loss_list, (1,0))
step_loss_list = np.transpose(step_loss_list, (1,0))

epoch_loss_list = epoch_loss_list[:int(epoch_loss_list.shape[0]*0.4),:]
step_loss_list = step_loss_list[:int(step_loss_list.shape[0]*0.4),:]

# print(epoch_loss_list, step_loss_list)
print(epoch_loss_list.shape, step_loss_list.shape)

label_name_list = ['w/o defense', 'MID 0.0', 'MID 1e-9', 'MID 1e-6', 'MID 0.001', 'MID 1', 'MID 100', 'MID 10000']
label_name_list = ['w/o defense', 'MID 0.0', 'MID 1e-9', 'MID 1e-6', 'MID 0.001', 'MID 1', 'MID 100']
# label_name_list = ['w/o defense', 'MID 0.0', 'MID 1e-9', 'MID 1e-6', 'MID 0.001', 'MID 1']
# label_name_list = ['w/o defense', 'MID 0.0', 'MID 1e-9', 'MID 1e-6']
# label_name_list = ['w/o defense', 'MID 0.0']

# plot epoch loss
fig, ax = plt.subplots()
# plt.yscale('log')
label_x = 'Epoch'
label_y = 'Loss'

# linestyle_str = [
#      ('solid', 'solid'),      # Same as (0, ()) or '-'
#      ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
#      ('dashed', 'dashed'),    # Same as '--'
#      ('dashdot', 'dashdot')]  # Same as '-.'

# linestyle_tuple = [
#      ('loosely dotted',        (0, (1, 10))),
#      ('dotted',                (0, (1, 1))),
#      ('densely dotted',        (0, (1, 1))),
#      ('long dash with offset', (5, (10, 3))),
#      ('loosely dashed',        (0, (5, 10))),
#      ('dashed',                (0, (5, 5))),
#      ('densely dashed',        (0, (5, 1))),

#      ('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      ('dashdotted',            (0, (3, 5, 1, 5))),
#      ('densely dashdotted',    (0, (3, 1, 1, 1))),

#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
line_style_list = [(0, (1, 1)), (0, (1, 3)), (0, (1, 5)), (5, (10, 3)), (0, (3, 2, 1, 2)), (0, (5, 5)), (0, (5, 1))]
for i in range(len(label_name_list)):
    ax.plot(np.arange(0,epoch_loss_list.shape[0],1),epoch_loss_list[:,i],label=label_name_list[i],linestyle=line_style_list[i])
# ax.plot(np.arange(0,epoch_loss_list.shape[0],1),epoch_loss_list,label=label_name_list)
ax.set_xlabel(label_x, fontsize=16)
ax.set_ylabel(label_y, fontsize=16)
# ax.set_xlabel(label_x, fontsize=16, fontdict={'family' : 'SimSun', 'weight':800})
# ax.set_ylabel(label_y, fontsize=16, fontdict={'family' : 'SimSun', 'weight':800})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim(0,8)
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig(f'{file_path_epoch}{mode}.png', dpi = 200)
plt.clf()

# plot step loss
fig, ax = plt.subplots()
# plt.yscale('log')
label_x = 'Step'
label_y = 'Loss'
for i in range(len(label_name_list)):
    ax.plot(np.arange(0,step_loss_list.shape[0],1),step_loss_list[:,i],label=label_name_list[i],linestyle=line_style_list[i])
# ax.plot(np.arange(0,step_loss_list.shape[0],1),step_loss_list,label=label_name_list)
ax.set_xlabel(label_x, fontsize=16)
ax.set_ylabel(label_y, fontsize=16)
# ax.set_xlabel(label_x, fontsize=16, fontdict={'family' : 'SimSun', 'weight':800})
# ax.set_ylabel(label_y, fontsize=16, fontdict={'family' : 'SimSun', 'weight':800})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim(0,8)
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig(f'{file_path_step}{mode}.png', dpi = 200)
plt.clf()