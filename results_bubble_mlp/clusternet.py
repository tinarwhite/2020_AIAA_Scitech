import tensorflow as tf
import numpy as np
from util import *
from util_tf import *
from time import time

starttime = time()

# Perceive training and test data.0) #angle = 6.0
#label, stride_t, stride_x, slope, angle = ('Fake Steps', 1, 1, 0.0, 20.0)
#label, stride_t, stride_x, slope, angle = ('Burgers', 20, 5, 0.0, 0.0)
label, stride_t, stride_x, slope, angle = ('Bubble', 1, 1, 0.0, 0.0)
state, scales = perceive(label, stride_t, stride_x, slope, angle)
n_io = (np.sum([i.shape[-1] for i in state[:-1]]), state[-1].shape[-1])

# Plot state desired
#stimuli, temporal_points, spatial_points, clips = state
#[plot_it(clips[i].T, spatial_points) for i in np.arange(clips.shape[0])]
#exit()

# Learning control variables
num_epochs = 0 #5000 #30000
plotting = True
learn_type = 'mlsp' # 'sft', 'init_f', 'init_c', 'segment', 'initialize', 'context' or 'mlsp' or 'mlp' with 1 n_clusters_network
learn_vars = 'Adam', 0.8, 0.8, 1.0*10**-10
#lr_vars = 'exp', 0.1, 0.96, 10000 #lr_decay_type, lr_start, lr_decay, lr_decay_steps #[['exp', 0.1, 0.001, num_epochs]] #[['poly', 0.1, 0.001, 10000]]
nonlin = ['relu','relu'] # [function, context] or [function, shape]
#n_hidden = [[40,40,40],[]]
batch_ratio = 0.005
n_clusters_network = 1
norm_type = 'L2'

n_hidden_runs = [[[40,40,40],[]]]
lr_vars_runs = [['by_loss', 0.1, 1, 1]]

for n_hidden in n_hidden_runs:
    for lr_vars in lr_vars_runs: 
        print(n_hidden, lr_vars)
        # Get save datafile name
        savename = get_savename(learn_type, n_hidden[0], norm_type, lr_vars[0], lr_vars[1], lr_vars[2])
        # Separate training data from test data
        chosen_stimuli, stimuli_range = (0,3)
        train_state = get_state_subset(chosen_stimuli, stimuli_range, state)
        # Network construction and training to convergence loop
        converged = False
        while converged == False:
            n_clusters = n_clusters_network
            # Build network
            network_vars = build_network(n_io, savename, n_hidden, n_clusters, learn_type, nonlin, learn_vars, lr_vars, train_state, norm_type)
            # Train network
            network_vars, total_error, num_epochs_print = train_network(network_vars, learn_vars, train_state, batch_ratio, num_epochs)
            # Check network for convergence criteria
            train_dict, n_clusters_network, learn_type, converged = check_convergence(savename, network_vars, n_clusters, train_state, learn_type)
            converged = True
        # Unpack network variables, calculate time taken and select time point and plot type for representative plotting 
        sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
        time_plot = 3 #65 or 4 for either model
        pflag = 'save'
        m, s = divmod(int(time()-starttime), 60)
        print('Training took %s minutes %s seconds'% (m,s))
        # Ploting function
        plot_cfd_results(savename, network_vars, state, time_plot, pflag)
        #plot_all(label, network_vars, state, n_clusters, stimuli_range, train_dict, time_plot, pflag, learn_type, num_epochs, num_epochs_print, total_error)

if num_epochs > 100: #5000
	plot_it(total_error[98:num_epochs_print])   

# Print out diagnostics
train_loss, test_loss = print_l2_loss(network_vars, state, train_dict)
print('Training L2 Loss, Test L2 Loss = ',(train_loss,test_loss))
print('Training took %s minutes %s seconds and found %s clusters'%(m,s,n_clusters_network))

# Troubleshooting
#print(sess.run(f,train_dict).shape) # f is (5, 15000)
#print(sess.run(c,train_dict).shape) #shape of c[0] = (3, 15000) ... c is (5, 3, 15000)
#print([v for v in tf.trainable_variables() if "s0" and "weight" in v.name])
#print(learn_type)
#print([v for v in tf.trainable_variables() if "weights_layer%s"%str(len(n_c_hidden)) in v.name and "context" in v.name])
QoIst, QoIsg = print_QoI_values(network_vars, state, scales[3])
