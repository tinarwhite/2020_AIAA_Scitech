import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime
from time import time
import pickle
import os

def perceive(label = 'Burgers', stride_t = 1, stride_x = 1, slope = 0.0, angle = 0.0):
    if label == 'Burgers':
       state, scales = import_burgers_data(stride_t, stride_x)
    if label == 'Bubble':
       state, scales = import_bubble_data(stride_t, stride_x)
    #stimuli, temporal_points, spatial_points, clips = state
    return state, scales

def get_state_subset(chosen_stimuli, stimuli_range, state):
    stimuli, temporal_points, spatial_points, clips = state
    return stimuli[chosen_stimuli:chosen_stimuli+stimuli_range], temporal_points, spatial_points, clips[chosen_stimuli:chosen_stimuli+stimuli_range]

def read_me(args):
    return np.hstack([np.loadtxt(x)[:,:500] for x in args])

def read_me_cfd(args):
    return np.stack([np.loadtxt(x) for x in args])

def import_bubble_data(stride_t = 1, stride_x = 1):
    # Define s
    #sall = np.array((1.4,1.8,2.0,3.0,5.0,9.0,20.0))
    s = np.array((1.4,2.0,5.0,1.8,3.0,9.0)) # Mach numbers
    ns = s.shape[0]
    # Code here for importing data from file
    input = ['bubble//cfd_results_M%s_gamma1.67_mu1.21e-2_lambda1.7e3.txt'%i for i in s]
    xy = read_me(['bubble//grid.txt']) # np, 2 where in this case np = nx*ny in form (nx,ny)
    #data = get_shock_bubble_data(input) # get all four state variables
    data = get_shock_bubble_data(input)[:,:,:,:1] #remove brackets to get all state variables back not just density
    print(data.shape)
    #plot_cfd_4(data[-1,-1]) # data shape is ns, nt, nx*ny, nv #also want to check if plotting is correct here
    # Define x, y, and t
    ns, nt, nxy, nv = data.shape
    s = s.reshape(ns,1)
    t = np.arange(nt).reshape(nt,1)
    raw_state = s, t, xy, data
    # Adjust scale and stride
    state, scales = scale_stride(raw_state, stride_t, stride_x)
    return state, scales

def scale_stride(raw_state, stride_t = 1, stride_x = 1):
    mu, t, x, output = raw_state
    # Scale to [-1,1]
    stimuli = np.divide((mu-np.min(mu)), (np.max(mu)-np.min(mu)))*2-1
    temporal_points = np.divide((t-np.min(t)), (np.max(t)-np.min(t)))*2-1
    spatial_points = np.divide((x-np.min(x)), (np.max(x)-np.min(x)))*2-1
    clips = np.divide((output-np.min(output)), (np.max(output)-np.min(output)))*2-1
    #clips = np.divide((output-np.min(output,(0,1,2))), (np.max(output,(0,1,2))-np.min(output,(0,1,2))))*2-1
    # Adjust stride
    clips = clips[:,::stride_t,::stride_x]
    spatial_points = spatial_points[::stride_x]
    temporal_points = temporal_points[::stride_t]
    state = stimuli, temporal_points, spatial_points, clips 
    scales = [[np.min(mu),np.max(mu)],[np.min(t),np.max(t)],[np.min(x),np.max(x)],[np.min(output),np.max(output)]]
    return state, scales

def get_shock_bubble_data(input):
    nv, nx, ny, nt = (4, 64, 32, 201) #number of variables, x points, y points, time points
    readme = read_me_cfd(input)
    print('Input shape is',readme.shape)
    data = readme[:,:,2:].reshape(len(input),nx*ny,nt,nv).transpose([0,2,1,3]) #to ns, nt, nx*ny , nv
    print('Input reshaped to',data.shape)
    return data

def plot_cfd_4(data, pflag = 'show', savelabel = '4'):
    #data shape must be nx*ny,nv 
    nx, ny = (64, 32)
    nv = data.shape[-1]
    data = data.reshape(nx,ny,nv).transpose([2,1,0]) #to get nv, ny, nx
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny)) 
    n_contours = 12
    if nv == 4:
        fig, axarr = plt.subplots(2,2,figsize=(20,10))
        for i in np.arange(data.shape[0])+1:
            plt.subplot(2,2,i)
            plt.contourf(X,Y,data[i-1],n_contours,cmap='jet')
            plt.subplot(2,2,i).set_xticklabels([])
            plt.subplot(2,2,i).set_yticklabels([])
    elif nv == 1:
        fix, ax = plt.subplots(1,figsize=(12,6))
        plt.contourf(X,Y,data[0],n_contours,cmap='jet')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0, hspace=0)
    # vars = ['density','pho_E','xmom','ymom'] #just to remember
    plt_save_or_show(plt, pflag, savelabel) # replaces plt.show()

def plt_save_or_show(plt, pflag, savelabel):
    if pflag == 'save':
        my_path =  os.getcwd() #os.path.dirname(os.path.realpath(__file__))
        plt.savefig(my_path + '//' + str(savelabel) + '.png') # datetime.now().strftime('%H.%M')
    elif pflag == 'show':
        plt.show()

def import_burgers_data(stride_t = 1, stride_x = 1):
    # Define mu
    mu = np.array((1,3,4,2,5))
    n_mu = mu.shape[0]
    # Code here for importing data from file
    output =  read_me(['burg/snaps_0p02_0p02_%s.dat'%m for m in mu]).T
    # Define x and t
    n_samp, n_x = output.shape
    n_t = int(n_samp/n_mu)
    mu = mu.reshape(n_mu,1)
    t = np.linspace(0.0, 500.0, n_t).reshape(n_t,1)
    x = np.linspace(0.0, 100.0, n_x).reshape(n_x,1)
    output = output.reshape(n_mu, n_t, n_x, 1)
    raw_state = mu, t, x, output
    # Adjust scale and stride
    state = scale_stride(raw_state, stride_t, stride_x)
    return state

def construct_fake_steps(stride_t = 1, stride_x = 1, slope = 0.0, angle = 0.0):
    # Construct fake input data
    n_mu, n_t, n_x = (3, 5, 250)
    t = np.linspace(1.0, 5.0, n_t)
    x = np.linspace(0.0, 100.0, n_x)
    mu = np.linspace(1.0, 5.0, n_mu)
    # Construct fake function data
    ind_center, ind_step0, ind_step1 = (int(n_x/2), int(n_x/10), int(n_x/5))
    ind_step05 = int((ind_step0+ind_step1)/2)
    yinit0, yinit1 = (10.0, 6.0)
    yinit05 = (yinit0+yinit1)/2
    yend0, yend1 = (2.0, 4.0)
    yend05 = (yend0+yend1)/2
    output = np.ones((n_mu*n_t,n_x))
    i = 0
    for indstep, yinit, yend in zip([ind_step0,ind_step05,ind_step1],[yinit0,yinit05,yinit1],[yend0,yend05,yend1]):
        for j in [2, 1, 0, -1, -2]:
            step_location = ind_center-indstep*j
            output[i,:step_location ] = yinit+x[:step_location ]*slope
            output[i, step_location:] = yend +x[ step_location:]*slope
            i += 1
    # Adjust scale and stride
    raw_state = mu, t, x, output.reshape(n_mu, n_t, n_x)
    stimuli, temporal_points, spatial_points, clips = scale_stride(raw_state, stride_t, stride_x)
    # Rotate outputs
    x1      = np.cos(angle*np.pi/180)*spatial_points.T - np.sin(angle*np.pi/180)*clips.reshape(n_mu*n_t, n_x)
    output1 = np.sin(angle*np.pi/180)*spatial_points.T + np.cos(angle*np.pi/180)*clips.reshape(n_mu*n_t, n_x)
    # Project back to the same x components
    output2 = np.zeros_like(output1)
    for i in range(output.shape[0]):
        output2[i] = np.interp(spatial_points,x1[i],output1[i])
    output2 = output2.reshape(n_mu, n_t, n_x, 1)
    stimuli = stimuli.reshape(n_mu,1)
    temporal_points = temporal_points.reshape(n_t,1)
    spatial_points = spatial_points.reshape(n_x,1)
    state = stimuli, temporal_points, spatial_points, output2
    return state

def write_file_list(the_filename,my_list):
    with open(the_filename, 'wb') as f:
        pickle.dump(my_list, f)

def read_file_list(the_filename):
    with open(the_filename, 'rb') as f:
        my_list_new = pickle.load(f)
    return my_list_new

def plot_it(v, v_indices = np.array([17]), i=0, flag = None, title = None, pltaxis = None):
    fig = plt.figure(figsize=(6,6)) #12,6
    plt.title(title,size=14)
    if (v_indices == np.array([17]))[0]:
         vspace = np.arange(v.shape[0])
         plt.plot(vspace, v, lw=2)
    else:
         vspace = v_indices
         plt.plot(vspace, v, lw=2)
    if pltaxis != None:
         plt.axis(pltaxis)
    if flag == 'save':
        my_path = 'C:\\Users\\Tina\\Desktop\\FRG\\NNs\\Project\\figs\\'
        plt.savefig(my_path + datetime.now().strftime('%H.%M') + '_' + str(i) + '_test.png')  
    else:
        plt.show()

def plot_two(v1, v1_indices, v2, v2_indices, i=0, flag = None, title = None, pltaxis = None):
    fig = plt.figure(figsize=(6,6)) #12,6
    plt.title(title,size=14)
    plt.plot(v1_indices, v1, lw=2)
    plt.plot(v2_indices, v2, 'ro',markersize=8)
    if pltaxis != None:
         plt.axis(pltaxis)
    if flag == 'save':
        my_path = 'C:\\Users\\Tina\\Desktop\\FRG\\NNs\\Project\\figs\\'
        plt.savefig(my_path + datetime.now().strftime('%H.%M') + '_' + str(i) + '_test.png')  
    else:
        plt.show()

def plot_only_function_network_results(network_vars, state, n_clusters, stimuli_range, train_dict, time_plot, pflag):
    stimuli, temporal_points, spatial_points, clips = state
    sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
    f_run = sess.run(f,train_dict)
    y__run = sess.run(y_,train_dict)
    f_run_plot = f_run.reshape(n_clusters,stimuli_range,temporal_points.shape[0],spatial_points.shape[0])
    argmin = np.argmin(np.abs(y__run-f_run),0)
    reconstructions = np.array([f_run[argmin[i],i] for i in range(f_run.shape[1])]).reshape(stimuli_range,temporal_points.shape[0],spatial_points.shape[0])
    plot_it(np.vstack((reconstructions[:,time_plot,:], clips[:,time_plot,:])).T,spatial_points[:,0],0,'training_data',pflag)
    [plot_it(np.vstack((f_run_plot[:,j,time_plot,:], clips[j,time_plot,:])).T,spatial_points[:,0],0,'training_data',pflag) for j in range(stimuli_range)]

def plot_segment_network_results(network_vars, state, n_clusters, stimuli_range, train_dict, time_plot, pflag):
    stimuli, temporal_points, spatial_points, clips = state
    sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
    f_run = sess.run(f,train_dict).reshape(n_clusters,stimuli_range,temporal_points.shape[0],spatial_points.shape[0])
    #f_run = sess.run(f[:-1],train_dict).reshape(n_clusters,stimuli_range,temporal_points.shape[0],spatial_points.shape[0])
    #c_run = sess.run(c,train_dict).reshape(n_clusters,stimuli_range,temporal_points.shape[0],spatial_points.shape[0])
    [plot_it(np.vstack((f_run[:,j,time_plot,:], clips[j,time_plot,:])).T,spatial_points[:,0],0,'training_data',pflag) for j in range(stimuli_range)]
    #[plot_it(np.vstack((c_run[:,j,time_plot,:], clips[j,time_plot,:])).T,spatial_points[:,0],0,'training_data',pflag) for j in range(stimuli_range)]
    #y__run = sess.run(y_,train_dict)
    #f_run_plot = f_run.reshape(n_clusters,stimuli_range,temporal_points.shape[0],spatial_points.shape[0])
    #argmin = np.argmin(np.abs(y__run-f_run),0)
    #reconstructions = np.array([f_run[argmin[i],i] for i in range(f_run.shape[1])]).reshape(stimuli_range,temporal_points.shape[0],spatial_points.shape[0])
    #plot_it(np.vstack((reconstructions[:,time_plot,:], clips[:,time_plot,:])).T,spatial_points[:,0],0,'training_data',pflag)
    #[plot_it(np.vstack((f_run_plot[:,j,time_plot,:], clips[j,time_plot,:])).T,spatial_points[:,0],0,'training_data',pflag) for j in range(stimuli_range)]

def plot_results(network_vars, state, time_plot, pflag):
    stimuli, temporal_points, spatial_points, clips = state
    sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
    batch_inputs, batch_ys = get_batch(1.0, state)
    total_dict = {input: batch_inputs, y_:batch_ys}
    print(sess.run(y, feed_dict=total_dict).shape)
    print((stimuli.shape[0],temporal_points.shape[0],spatial_points.shape[0]))
    predictions = sess.run(y, feed_dict=total_dict).reshape(stimuli.shape[0],temporal_points.shape[0],spatial_points.shape[0])
    plot_it(np.vstack((predictions[:,time_plot], clips[:,time_plot,:,0])).T,spatial_points[:,0],0,'training_data',pflag)
    plot_it(np.vstack((predictions[:,time_plot])).T,spatial_points[:,0],0,'training_data',pflag)
    [plot_it(np.vstack((predictions[list(stimuli).index(stimulus),:],  clips[list(stimuli).index(stimulus),:,:,0])).T,spatial_points[:,0],0,'training_data_%s' % stimulus,pflag) for stimulus in stimuli]

def plot_all(label, network_vars, state, n_clusters, stimuli_range, train_dict, time_plot, pflag, learn_type, num_epochs, num_epochs_print, total_error):
    if label == 'Bubble':
        plot_cfd_results(network_vars, state, time_plot, pflag)
    else:
        if learn_type == 'initialize' or learn_type == 'context':
            plot_only_function_network_results(network_vars, state, n_clusters, stimuli_range, train_dict, time_plot, pflag)
        if learn_type == 'segment' or learn_type == 'init_c' or learn_type == 'init_f' or learn_type == 'sft2' or learn_type == 'sft3':
            plot_segment_network_results(network_vars, state, n_clusters, stimuli_range, train_dict, time_plot, pflag)
        # Plot after training both networks
        plot_results(network_vars, state, time_plot, pflag)
    if num_epochs > 12: #5000
        plot_it(total_error[9:num_epochs_print])    

def get_batch(batch_ratio, state):
    stimuli, temporal_points, spatial_points, clips = state
    indices_train = np.nonzero(np.ones((stimuli.shape[0], temporal_points.shape[0], spatial_points.shape[0])))
    if batch_ratio == 1.0:
        batch_choices = np.arange(0,indices_train[0].shape[0])
    else:
        batch_size = np.int(indices_train[0].shape[0]*batch_ratio)
        batch_choices = np.random.random_integers(0,indices_train[0].shape[0]-1, batch_size) # random batches
    batch_indices = [indices_train[i][batch_choices] for i in np.arange(3)]
    batch_inputs = np.hstack((stimuli[batch_indices[0]], temporal_points[batch_indices[1]], spatial_points[batch_indices[2]])).T
    batch_ys = clips[batch_indices].T
    return batch_inputs, batch_ys

def plot_cfd_results(savename, network_vars, state, time_plot, pflag):
    stimuli, temporal_points, spatial_points, clips = state
    sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
    batch_inputs, batch_ys = get_batch(1.0, state)
    total_dict = {input: batch_inputs, y_:batch_ys}
    truth = sess.run(y_, feed_dict=total_dict).T.reshape(stimuli.shape[0],temporal_points.shape[0],spatial_points.shape[0],clips.shape[-1])
    predictions = sess.run(y, feed_dict=total_dict).T.reshape(stimuli.shape[0],temporal_points.shape[0],spatial_points.shape[0],clips.shape[-1])
    #plot_cfd_4(clips[-2,-2],pflag,'truth_clips') # in ns, nt, nxy, nv
    #plot_cfd_4(truth[-1,-1],pflag,'plt_truth') #[-3,-2] for test set
    #plot_cfd_4(predictions[-1,-1],pflag,'plt_guess_'+savename)
    for i in np.arange(stimuli.shape[0]):
        plot_cfd_4(truth[i,-1],pflag,'plt_'+str(i)+'_truth') #[-3,-2] for test set
        plot_cfd_4(predictions[i,-1],pflag,'plt_'+str(i)+'_guess_'+savename)
        plot_cfd_4(np.square(predictions[i,-1]-truth[i,-1]),pflag,'plt_'+str(i)+'_error_'+savename)

def print_l2_loss(network_vars, state, train_dict):
    sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
    train_loss = sess.run(tf.sqrt(tf.reduce_mean(tf.square(y_-y))),train_dict)
    chosen_stimuli, stimuli_range = (2,1)
    test_state = get_state_subset(chosen_stimuli, stimuli_range, state)
    batch_inputs, batch_ys = get_batch(1.0, test_state)
    test_dict = {input: batch_inputs, y_:batch_ys}
    test_loss = sess.run(tf.sqrt(tf.reduce_mean(tf.square(y_-y))),test_dict)
    return train_loss, test_loss

def print_QoI_values(network_vars, state, scale):
    sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
    chosen_stimuli, stimuli_range = (0,5)
    state = get_state_subset(chosen_stimuli, stimuli_range, state)
    batch_inputs, batch_ys = get_batch(1.0, state)
    all_dict = {input: batch_inputs, y_:batch_ys}
    truth = sess.run(y_,all_dict).reshape(state[3].shape[:-1])
    guesses = sess.run(y,all_dict).reshape(state[3].shape[:-1])
    q0t, q0g = unscale(truth[:,150,400],scale), unscale(guesses[3:,150,400], scale)
    q1t, q1g = unscale(np.mean(truth[:,50,:],1),scale), unscale(np.mean(guesses[3:,50,:],1), scale)
    q2t, q2g = unscale(np.mean(truth[:,:,200],1),scale), unscale(np.mean(guesses[3:,:,200],1), scale)
    q3t, q3g = unscale(np.mean(truth[:,:,:],(1,2)),scale), unscale(np.mean(guesses[3:,:,:],(1,2)), scale)
    QoIst = np.array([q0t, q1t, q2t, q3t]) #, q2, q3, q4
    QoIsg = np.array([q0g, q1g, q2g, q3g]) #, q2, q3, q4
    #print('QoIst =',QoIst)
    #print('QoIsg =',QoIsg)
    #Pct_Error = np.mean(np.divide(np.abs(unscale(truth[3:,:,:],scale)-unscale(guesses[3:,:,:],scale)),np.mean(np.abs(unscale(truth[3:],scale)))))*100 #% Error
    Pct_Error = np.mean(np.abs(unscale(truth[3:,:,:],scale)-unscale(guesses[3:,:,:],scale))/0.00511911)*100 #% Error
    QoI_Pct_Error = calc_QoI_Pct_Error(QoIst, QoIsg)
    print('Pct_Error =',Pct_Error)
    print('QoI_Pct_Error =',QoI_Pct_Error)
    SF_QoI_Pct_Error = [0,0,0,0]
    for order in [0,1,2,3]:
        sftime = time()
        SF_QoIsg = surface_fitting(QoIst,order)
        print('Surface fitting order %s took %s seconds'%(order,time()-sftime))
        SF_QoI_Pct_Error[order] = calc_QoI_Pct_Error(QoIst,SF_QoIsg)
    print('SF_QoIsg =',SF_QoIsg)
    print('SF_QoI_Pct_Error =',SF_QoI_Pct_Error)
    return QoIst, QoIsg

def surface_fitting(QoIst,order):
    p = [np.poly1d(np.polyfit([1,3,4],fy, order)) for fy in QoIst[:,:3]]
    SF_QoIsg = np.array([pi((2,5)) for pi in p])
    #print('SF_QoIsg =',SF_QoIsg)
    return SF_QoIsg

def calc_QoI_Pct_Error(QoIst, QoIsg):
    #return np.mean(np.divide(np.abs(QoIst[:,-2:]-QoIsg),np.mean(np.abs(QoIst[:,-2:]))))*100
    return np.mean(np.divide(np.abs(QoIst[:,-2:]-QoIsg),0.00511911))*100
    
def unscale(scaled_value, scale):
    unscaled_value = scale[0]+(scale[1]-scale[0])*(scaled_value+1.0)/2.0
    return unscaled_value
