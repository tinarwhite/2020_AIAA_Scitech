import tensorflow as tf
import numpy as np
from cutil import *
from cutil_tf import *
from time import time

starttime = time()

# Perceive training and test data.0) #angle = 6.0
#label, stride_t, stride_x, slope, angle = ('Affine', 1, 1, 0.0, 0.0)
#label, stride_t, stride_x, slope, angle = ('Identity', 1, 1, 0.0, 0.0)
#label, stride_t, stride_x, slope, angle = ('Reverse', 1, 1, 0.0, 0.0)
#label, stride_t, stride_x, slope, angle = ('Fake Steps', 1, 1, 0.0, 20.0)
label, stride_t, stride_x, slope, angle = ('Burgers', 1, 1, 0.0, 0.0)
#label, stride_t, stride_x, slope, angle = ('Bubble', 2, 1, 0.0, 0.0)
state, scales = perceive(label, stride_t, stride_x, slope, angle)
n_io = (np.sum([i.shape[-1] for i in state[:-1]]), state[-1].shape[-1])

# Plot state desired
#stimuli, temporal_points, spatial_points, clips = state
#[plot_it(clips[i,:,:,0].T, spatial_points) for i in np.arange(clips.shape[0])]
#exit()

# Learning control variables
num_epochs = 0 #5000 #30000
plotting = True
pflag = 'save'
learn_type = 'mlp' # 'sft', 'init_f', 'init_c', 'segment', 'initialize', 'context' or 'mlsp' or 'mlp' or 'mlpsc' or 'mlpec' or 'resnet'
learn_vars = 'RMSProp', 0.8, 0.8, 1.0*10**-12
#lr_vars = 'exp', 0.1, 0.96, 10000 #lr_decay_type, lr_start, lr_decay, lr_decay_steps #[['exp', 0.1, 0.001, num_epochs]] #[['poly', 0.1, 0.001, 10000]]
nonlin = ['tanh','tanh'] # [function, context] or [function, shape]
#n_hidden = [[40,40,40],[]]
batch_ratio = 0.0002
n_clusters_network = 2
norm_type = 'L2'

n_hidden_runs = [[[20]*4,[4]*3]]
#lr_vars_runs = [['by_loss', 1.0, 1, 1]] #[['exp', 0.0001, 1.0, 10000]]  #[['exp', 0.01, 0.1, 10000]] #[['by_loss', 1.0, 1, 1]]
lr_vars_runs = [['exp', 0.001, 1.0, num_epochs]] #[['exp', 0.1, 0.0002, num_epochs]] good one

for n_hidden in n_hidden_runs:
    for lr_vars in lr_vars_runs: 
        print(n_hidden, lr_vars)
        # Get save datafile name
        savename = get_savename(learn_type, n_hidden[0], norm_type, lr_vars[0], lr_vars[1], lr_vars[2])
        # Separate training data from test data
        chosen_stimuli, stimuli_range = (0,3)
        train_state = get_state_subset(chosen_stimuli, stimuli_range, state)
        test_state = get_state_subset(3, 2, state)
        print(train_state[3].shape)
        # Network construction and training to convergence loop
        converged = False
        while converged == False:
            n_clusters = n_clusters_network
            # Build network
            network_vars = build_network(n_io, savename, n_hidden, n_clusters, learn_type, nonlin, learn_vars, lr_vars, train_state, norm_type)
            # Train network
            network_vars, total_error, total_weights, num_epochs_print = train_network(network_vars, learn_vars, train_state, test_state, batch_ratio, num_epochs, norm_type)
            # Check network for convergence criteria
            train_dict, n_clusters_network, learn_type, converged = check_convergence(savename, network_vars, n_clusters, train_state, learn_type, n_hidden)
            converged = True
        # Unpack network variables, calculate time taken and select time point and plot type for representative plotting 
        sess, input, y_, f, c, y, loss_runs, reg_loss, train_step_runs, learning_rate = network_vars
        time_plot = 100 #65 or 4 for either model
        m, s = divmod(int(time()-starttime), 60)
        print('Training took %s minutes %s seconds'% (m,s))
        # Ploting function
        #plot_cfd_results(savename, network_vars, state, time_plot, pflag)
        plot_all(savename, label, network_vars, state, n_clusters, stimuli_range, train_dict, time_plot, pflag, learn_type, num_epochs, num_epochs_print, total_error) if plotting == True else None

start_i = 198 if num_epochs > 200 else 0
#plot_it(total_error[start_i:num_epochs_print],flag='save', savename='plt_error')   
#plot_it(total_weights[start_i:num_epochs_print],flag='save', savename='plt_weights')   

# Print out diagnostics
train_loss, test_loss = print_l2_loss(network_vars, state, train_dict)
print('Training L2 Loss, Test L2 Loss = ',(train_loss,test_loss))
print('Training took %s minutes %s seconds and found %s clusters'%(m,s,n_clusters_network))

# Troubleshooting
#print(sess.run(f,train_dict).shape) # f is (5, 15000)
#print(sess.run(c,train_dict).shape) #shape of c[0] = (3, 15000) ... c is (5, 3, 15000)
#print([v for v in tf.trainable_variables() if "s0" and "weight" in v.name])
print([v for v in tf.trainable_variables() if "context" in v.name])
print([v for v in tf.trainable_variables() if "context" in v.name and "biases" not in v.name])
#print(learn_type)
#print([v for v in tf.trainable_variables() if "weights_layer%s"%str(len(n_c_hidden)) in v.name and "context" in v.name])
#print("s0")
#print([v for v in tf.trainable_variables() if "s0" in v.name and "biases" not in v.name])
#all_weights, _ = sess.run(tf_flat_vars(tf.trainable_variables()))
#print(all_weights.shape)
#print(np.sum(np.less(np.abs(all_weights),0.01)),np.sum(np.less(0.01,np.abs(all_weights))))

#print(state[3].shape)
QoIst, QoIsg = print_QoI_values(network_vars, state, scales[3])
#print(scales)
#average u, average u at some t, average u at some x, random u value at some x, t

