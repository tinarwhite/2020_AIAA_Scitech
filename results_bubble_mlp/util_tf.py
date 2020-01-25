import tensorflow as tf
import numpy as np
import os
from scipy.stats import linregress
from util import *
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_network(n_io, savename, n_hidden, n_clusters, learn_type, nonlin, learn_vars, lr_vars, train_state, norm_type):
    # Construct network architecture
    input, y_, y, c, f = network_architecture(n_io, n_clusters, n_hidden[0], n_hidden[1], learn_type, nonlin)
    # Define a loss function
    loss_runs = define_loss(y_, y, c, f, learn_type, train_state, n_clusters, norm_type)
    # Choose trainable variables
    train_var_lists = choose_trainable_variables(learn_type)
    # Choose an optimization procedure
    train_step_runs, learning_rate = choose_optimization(loss_runs, train_var_lists, learn_vars, lr_vars)
    # Load memory if exists and initialize network
    sess = initialize_network(savename,learn_type, train_state, input, y, y_, f, c)
    return sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate

def network_architecture(n_io, n_clusters,n_hidden,n_c_hidden,learn_type,nonlin):
    tf.reset_default_graph()
    input = tf.placeholder("float32", [n_io[0], None])
    y_ = tf.placeholder("float32", [n_io[1], None])
    if learn_type == 'mlp': 
        print('Building mlp network architecture...')
        c, f = (tf.ones_like(y_), y_) # dummy values
        y = single_mlp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0])
    if learn_type == 'mlsp': 
        print('Building mlsp network architecture...')
        c, f = (tf.ones_like(y_), y_) # dummy values
        y = single_mlsp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], mult = 1.0)
    if learn_type == 'segment' or learn_type == 'init_f' or learn_type == 'init_c':
        print('Building segment network architecture...')
        c, f = (tf.ones_like(y_), y_) # dummy values
        y, c, f = segment_architecture(n_io, input,n_clusters,n_hidden,n_c_hidden,learn_type,nonlin)
    if learn_type == 'initialize' or learn_type == 'cluster':
        print('Building clustered network architecture...')
        y, c, f = cluster_architecture(n_io, input,n_clusters,n_hidden,n_c_hidden,learn_type,nonlin)
    if learn_type == 'sft' or learn_type == 'sft2' or learn_type == 'sft3':
        print('Building', learn_type, 'network architecture...')
        c, f = (tf.ones_like(y_), y_) # dummy values
        y, c, f = sft_architecture(n_io, input,n_clusters,n_hidden,n_c_hidden,learn_type,nonlin)
    return input, y_, y, c, f

def sft_architecture(n_io, input, n_clusters,n_hidden,n_s_hidden,learn_type,nonlin):
    shape_stack = [0] * n_clusters
    function_stack = [0] * n_clusters
    transition_stack = [0] * n_clusters
    for n in range(n_clusters):
        with tf.variable_scope("shape%s"% str(n)):
            shape_stack[n] = single_mlsp_network(input, [n_io[0]] + n_s_hidden + [n_io[0]], nonlin[1], nonlin_final = 'linear')
        with tf.variable_scope("function%s"% str(n)):
            function_stack[n] = single_mlsp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') 
        #with tf.variable_scope("transition%s"% str(n)):
        #    transition_stack[n] = single_mlsp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') #shape_stack[n]
    shapes = tf.stack(shape_stack,0)
    functions = tf.concat(function_stack,0)
    #transitions = tf.concat(transition_stack,0)
    with tf.variable_scope("transition"):
        #shapes_flattened = tf.transpose(tf.contrib.layers.flatten(tf.transpose(shapes,[2,1,0])),[1,0])
        #transition_input = tf.concat((shapes_flattened,functions),0)
        #transitions = single_mlsp_network(transition_input, [4*n_clusters] + n_s_hidden + [n_io[1]], nonlin[1], nonlin_final = 'linear')
        #transition_input = tf.concat((input,functions),0)
        transition_input = input
        transitions = single_mlsp_network(transition_input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[1], nonlin_final = 'linear')
    shape_conditions, assigned_conditions, assigned_conditions_untiled = get_sft_conditions(shapes, n_clusters)
    contexts = tf.where(shape_conditions,tf.ones_like(functions),tf.zeros_like(functions))
    #states = tf.where(assigned_conditions, tf.multiply(functions,contexts), tf.multiply(functions,transitions))
    states = tf.where(assigned_conditions, tf.multiply(functions,contexts), tf.tile(transitions,[n_clusters,1]))
    guess = tf.reduce_sum(states,0,keep_dims=True)
    return guess, [shapes, contexts, states], functions #y/g, c, f

def get_sft_conditions(shapes, n_clusters):
    shape_conditions = get_shape_conditions_reduced_c3(shapes)
    assigned_conditions_untiled = tf.logical_not(tf.equal(tf.count_nonzero(shape_conditions,0,True),0)) 
    assigned_conditions = tf.tile(assigned_conditions_untiled,[n_clusters,1]) # True across all clusters if at least one cluster matches perfectly
    return shape_conditions, assigned_conditions, assigned_conditions_untiled

def sft_architecture_unfinished_opt1(n_io, input, n_clusters,n_hidden,n_s_hidden,learn_type,nonlin):
    shape_stack = [0] * n_clusters
    function_stack = [0] * n_clusters
    for n in range(n_clusters):
        with tf.variable_scope("shape%s"% str(n)):
            shape_stack[n] = single_mlsp_network(input, [n_io[0]] + n_s_hidden + [n_io[0]], nonlin[1], nonlin_final = 'linear')
        with tf.variable_scope("function%s"% str(n)):
            function_stack[n] = single_mlsp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') 
    shapes = tf.stack(shape_stack,0)
    functions = tf.concat(function_stack,0)
    with tf.variable_scope("transition"):
        shapes_flattened = tf.transpose(tf.contrib.layers.flatten(tf.transpose(shapes,[2,1,0])),[1,0])
        transition_input = tf.concat((shapes_flattened,functions),0)
        transitions = single_mlsp_network(transition_input, [4*n_clusters] + n_s_hidden + [n_io[1]], nonlin[1], nonlin_final = 'linear')
    guess = tf.reduce_sum(states,0,keep_dims=True)
    return guess, shapes, functions #y/g, c, f

def segment_architecture(n_io, input, n_clusters,n_hidden,n_s_hidden,learn_type,nonlin):
    shape_stack = [0] * n_clusters
    function_stack = [0] * n_clusters
    for n in range(n_clusters):
        with tf.variable_scope("shape%s"% str(n)):
            shape_stack[n] = single_mlsp_network(input, [n_io[0]] + n_s_hidden + [n_io[1]], nonlin[1], nonlin_final = 'linear')
        with tf.variable_scope("function%s"% str(n)):
            function_stack[n] = single_mlsp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') 
    shapes = tf.concat(shape_stack,0)
    functions = tf.concat(function_stack,0)
    states = tf.multiply(functions,tf.maximum(tf.minimum(shapes,1),0))
    guess = tf.reduce_sum(states,0,keep_dims=True)
    return guess, shapes, functions #y/g, c, f

def segment_architecture_good(n_io, input, n_clusters,n_hidden,n_s_hidden,learn_type,nonlin):
    shape_stack = [0] * n_clusters
    function_stack = [0] * n_clusters
    for n in range(n_clusters):
        with tf.variable_scope("shape%s"% str(n)):
            shape_layer = single_mlsp_network(input, [n_io[0]] + n_s_hidden + [n_io[0]], nonlin[1], nonlin_final = 'linear')
        with tf.variable_scope("function%s"% str(n)):
            function_layer = single_mlsp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') 
        function_stack[n] = function_layer
        shape_stack[n] = shape_layer #shape_layer[2:]
    shapes = tf.stack(shape_stack,0)
    functions_orig = tf.concat(function_stack,0)
    conditions = get_closest_shape_conditions(shapes, n_clusters)
    states = tf.where(conditions, functions_orig, tf.zeros_like(functions_orig))
    guess = tf.reduce_sum(states,0,keep_dims=True)
    return guess, shapes, functions_orig #y/g, c, f

def segment_architecture_tran1(n_io, input, n_clusters,n_hidden,n_s_hidden,learn_type,nonlin):
    shape_stack = [0] * n_clusters
    function_stack = [0] * (n_clusters-1)
    for n in range(n_clusters):
        #with tf.variable_scope("zoom%s"% str(n)):
            #zoom_layer = single_mlp_network(input, [n_io[0]] + [n_io[0]], nonlin = 'linear', nonlin_final = 'linear')
        with tf.variable_scope("shape%s"% str(n)):
            shape_layer = single_mlsp_network(input, [n_io[0]] + n_s_hidden + [n_io[0]], nonlin[1], nonlin_final = 'linear')
        if n != n_clusters-1:
            with tf.variable_scope("function%s"% str(n)):
                function_layer = single_mlsp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') 
            function_stack[n] = function_layer
        shape_stack[n] = shape_layer #shape_layer[2:]
    shapes = tf.stack(shape_stack,0)
    shapes_c1 = tf.reduce_sum(shapes,1)
    functions = tf.concat(function_stack,0)
    with tf.variable_scope("function_transition"):
        transition_function = single_mlsp_network(functions, [n_clusters-1] + [n_clusters-1] + [n_io[1]], nonlin[1], nonlin_final = 'linear')
    functions = tf.concat((functions,transition_function),0)
    conditions = get_shape_conditions_reduced_c3(shapes)
    #conditions = get_closest_shape_conditions(shapes, n_clusters)
    states = tf.where(conditions, functions, tf.tile(transition_function,[n_clusters, 1]))
    #states = tf.multiply(tf.nn.softmax(shapes_c1,0), functions)
    guess = tf.reduce_sum(states,0,keep_dims=True)
    return guess, shapes, functions #y/g, c, f

def segment_architecture_c1(n_io, input, n_clusters,n_hidden,n_s_hidden,learn_type,nonlin):
    shape_stack = [0] * n_clusters
    function_stack = [0] * n_clusters
    for n in range(n_clusters):
        #with tf.variable_scope("zoom%s"% str(n)):
            #zoom_layer = single_mlp_network(input, [n_io[0]] + [n_io[0]], nonlin = 'linear', nonlin_final = 'linear')
        with tf.variable_scope("shape%s"% str(n)):
            shape_layer = single_mlp_network(input, [n_io[0]] + n_s_hidden + [n_io[1]], nonlin[1], nonlin_final = 'linear')
        with tf.variable_scope("function%s"% str(n)):
            function_layer = single_mlp_network(shape_layer, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') 
        shape_stack[n] = shape_layer #shape_layer[2:]
        function_stack[n] = function_layer
    shapes = tf.concat(shape_stack,0)
    functions = tf.concat(function_stack,0)
    conditions = get_shape_conditions(shapes)
    states = tf.where(conditions, functions, tf.zeros_like(functions))
    guess = tf.reduce_sum(states,0,keep_dims=True)
    return guess, shapes, functions #y/g, c, f

def get_shape_conditions(shape, val = [1.0,1.0]):
    max1, min1 = (val[0]*tf.constant([1.0],dtype=tf.float32), val[1]*tf.constant([-1.0],dtype=tf.float32))
    condition = tf.logical_and(tf.less(shape, max1), tf.less(min1, shape))
    return condition

def get_shape_conditions_reduced_c3(shape, val = [1.0,1.0]):
    max1, min1 = (val[0]*tf.constant([[1.0],[1.0],[1.0]],dtype=tf.float32), val[1]*tf.constant([[-1.0],[-1.0],[-1.0]],dtype=tf.float32))
    p_condition = tf.logical_and(tf.less(shape, max1), tf.less(min1, shape))
    p_condition_01 = tf.logical_and(p_condition[:,0], p_condition[:,1])
    condition = tf.logical_and(p_condition_01, p_condition[:,2])
    return condition

def get_shape_conditions_c3(shape, val = [1.0,1.0]):
    max1, min1 = (val[0]*tf.constant([[1.0],[1.0],[1.0]],dtype=tf.float32), val[1]*tf.constant([[-1.0],[-1.0],[-1.0]],dtype=tf.float32))
    condition = tf.logical_and(tf.less(shape, max1), tf.less(min1, shape))
    return condition

def get_closest_shape_conditions(shapes, n_clusters):
    shape_distance = tf.reduce_sum(tf.abs(shapes), 1)
    min_distance_tiled = tf.tile(tf.reduce_min(shape_distance,0, keep_dims = True), [n_clusters, 1])
    condition = tf.equal(min_distance_tiled, shape_distance)
    return condition

def segment_architecture_old(n_io, input, n_clusters,n_hidden,n_s_hidden,learn_type,nonlin):
    transition_stack = [0] * n_clusters
    interior_stack = [0] * n_clusters
    for n in range(n_clusters):
        with tf.variable_scope("scope%s"% str(n)):
            scope_layer = single_mlp_network(input, [n_io[0]] + [n_io[0]], nonlin = 'linear', nonlin_final = 'linear')
        with tf.variable_scope("shape%s"% str(n)):
            shape_layer = single_mlp_network(scope_layer, [n_io[0]] + n_s_hidden + [n_io[0]], nonlin[1], nonlin_final = 'linear')
        with tf.variable_scope("transition%s"% str(n)):
            transition_layer = single_mlp_network(shape_layer, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') 
        with tf.variable_scope("interior%s"% str(n)):
            interior_layer = single_mlp_network(shape_layer, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0], nonlin_final = 'linear') 
        current_state = tf.zeros_like(interior_layer)
        #current_state = conditionals(current_state, shape_layer, interior_layer, 'interior')
        current_state = conditionals(current_state, shape_layer, transition_layer, 'transition')
        interior_stack[n] = interior_layer
        transition_stack[n] = transition_layer
    return current_state, tf.concat(transition_stack,0), tf.concat(interior_stack,0) #y, c, f

def cluster_architecture(n_io, input, n_clusters,n_hidden,n_c_hidden,learn_type,nonlin):
    functions = ["function%s"% str(n) for n in range(n_clusters)]
    contexts = ["context%s"% str(n) for n in range(n_clusters)]
    function_layers = [0] * n_clusters
    context_layers = [0] * n_clusters
    for c, function in enumerate(functions):
        with tf.variable_scope(function):
            function_layers[c] = single_mlp_network(input, [n_io[0]] + n_hidden + [n_io[1]], nonlin[0]) #switch
    for c, context in enumerate(contexts):
        with tf.variable_scope(context):
            context_layers[c] = single_mlp_network(input, [n_io[0]] + n_c_hidden + [n_io[1]], nonlin[1], nonlin_final = 'tanh') #switch # nonlin_final = 'tanh'
    if nonlin[1] == 'linear':
        c = tf.clip_by_value(tf.concat(context_layers,0), 0.0, 1.0)
        #c = tf.concat(context_layers,0)
    else:
        c = tf.nn.softmax(tf.concat(context_layers,0),0)
        #c = tf.concat(context_layers,0)
    f = tf.concat(function_layers,0)
    fc = tf.multiply(c, f)
    y = tf.reduce_sum(fc,0,keep_dims=True)
    return y, c, f

def single_mlsp_network(input, n_hidden, nonlin = 'tanh', nonlin_final = 'linear', mult = 0.01):
    out = single_mlp_network(input, n_hidden[:1] + n_hidden[-1:], nonlin, nonlin_final, 0, mult)
    for i in range(len(n_hidden)-2):
        out += single_mlp_network(input, n_hidden[:1+i+1] + n_hidden[-1:], nonlin, nonlin_final, i+1, mult)
    return out

def single_mlp_network(input, n_hidden, nonlin = 'tanh', nonlin_final = 'linear', simp = 0, mult = 1.0):
    mlp_initializer = tf.random_uniform_initializer(mult*-1.0,1.0*mult)
    hidden_layer = input
    size_progression = n_hidden
    for i, size in enumerate(zip(size_progression[:-1], size_progression[1:])):
        weights = tf.get_variable("weights_s%s_layer%s"%(simp,str(i)),(size[1],size[0]),initializer=mlp_initializer)
        biases = tf.get_variable("biases_s%s_layer%s"%(simp,str(i)), (size[1], 1),initializer=mlp_initializer)
        if i != len(size_progression)-2:
            nonlinearity = nonlin
        if i == len(size_progression)-2: 
            nonlinearity = nonlin_final
        hidden_layer = apply_nonlinearity(hidden_layer, weights, biases, nonlinearity)
    return hidden_layer

def apply_nonlinearity(hidden_layer, weights, biases, nonlinearity):
    if nonlinearity == 'linear':
        hidden_layer = tf.matmul(weights,hidden_layer) + biases
    if nonlinearity == 'tanh':
        hidden_layer = tf.nn.tanh(tf.matmul(weights,hidden_layer) + biases)
    if nonlinearity == 'relu':
        hidden_layer = tf.nn.relu(tf.matmul(weights,hidden_layer) + biases)
    if nonlinearity == 'relu6':
        hidden_layer = tf.nn.relu6(tf.matmul(weights,hidden_layer) + biases)
    if nonlinearity == 'leaky_relu':
        hidden_layer = tf.nn.leaky_relu(tf.matmul(weights,hidden_layer) + biases)
    if nonlinearity == 'sigmoid':
        hidden_layer = tf.nn.sigmoid(tf.matmul(weights,hidden_layer) + biases)    
    return hidden_layer

def choose_optimization(loss_runs, train_var_lists, learn_vars, lr_vars):
    # Apply exponential decay to learning rate
    optim, decay, mom, _ = learn_vars
    lr_decay_type, lr_start, lr_decay, lr_decay_steps = lr_vars  
    global_step = tf.get_variable('global_step', shape=[], initializer=tf.zeros_initializer(), trainable=False)
    if lr_decay_type == 'no decay':
        learning_rate = [tf.constant(lr_start)]*len(loss_runs)
    elif lr_decay_type == 'exp':
        learning_rate = [tf.train.exponential_decay(lr_start, global_step, lr_decay_steps, lr_decay, staircase=False)]*len(loss_runs)
    elif lr_decay_type == 'poly':
        end_learning_rate = lr_decay
        power = 0.5
        learning_rate = [tf.train.polynomial_decay(lr_start, global_step, lr_decay_steps, end_learning_rate, power)]*len(loss_runs)
    elif lr_decay_type == 'by_loss':
        learning_rate = [tf.minimum(loss_run*lr_start,0.05) for loss_run in loss_runs]
    elif lr_decay_type == 'by_loss2':
        learning_rate = [tf.minimum(loss_run*loss_run*lr_start,0.05) for loss_run in loss_runs]
    # Use Adam or RMSProp
    if optim == 'RMSProp':
        train_step_runs = [tf.train.RMSPropOptimizer(lr, decay, mom).minimize(loss_run, var_list = train_var_list, global_step=global_step) for lr, loss_run, train_var_list in zip(learning_rate,loss_runs,train_var_lists)]
    if optim == 'Adam':
        train_step_runs = [tf.train.AdamOptimizer(lr).minimize(loss_run, var_list = train_var_list, global_step=global_step) for lr, loss_run, train_var_list in zip(learning_rate,loss_runs,train_var_lists)]
    return train_step_runs, learning_rate 

def tf_get_norm(val, type):
    if type == 'L1':
        norm = tf.reduce_mean(tf.abs(val))
    if type == 'L2':
        norm = tf.sqrt(tf.reduce_mean(tf.square(val)))
    if type == 'L2_noroot_scaled_by5':
        norm = tf.reduce_mean(tf.square(val))
    return norm

def define_loss(y_, y, c, f, learn_type, train_state, n_clusters, norm_type):
    # define loss and training variables depending on which run it is, with mlp as the baseline
    reg_loss = define_regularization_loss(learn_type, 0.0001)
    #loss_deriv = define_loss_deriv(y_,y, np.array((train_state[4].shape)), train_loss_deriv = True)
    loss_min = tf.reduce_mean(tf.reduce_min(tf.abs(y_-f),0)) + reg_loss
    loss_guess = tf_get_norm(y_-y, norm_type) + reg_loss #+ loss_deriv
    loss_runs = [loss_guess]
    if learn_type == 'initialize': 
        loss_runs = [loss_min]
    if learn_type == 'init_f':
        loss_runs = [loss_min]
    if learn_type == 'init_c':
        loss_ctol = get_loss_ctol(n_clusters, y_, f, c) #+ reg_loss
        loss_runs = [loss_ctol]
    if learn_type == 'segment': 
        loss_ctol = get_loss_ctol(n_clusters, y_, f, c) #+ reg_loss
        loss_runs = [loss_min, loss_guess+loss_ctol*100.0]
    if learn_type == 'sft' or learn_type == 'sft2' or learn_type == 'sft3': #['transition','function','shape']
        shapes, contexts, states = c
        shape_conditions, assigned_conditions, assigned_conditions_untiled = get_sft_conditions(shapes, n_clusters)  
        fun_vals = tf.abs(tf.multiply(y_,tf.reduce_sum(contexts,0,True))-tf.reduce_sum(tf.multiply(f,contexts),0,True))  
        loss_fun = tf.divide(tf.reduce_mean(fun_vals),tf.count_nonzero(fun_vals,dtype = tf.float32)) + reg_loss
        #loss_runs = [loss_guess, loss_fun*100.0, loss_stol*100.0]
        if learn_type == 'sft':
            loss_stol = get_loss_stol(n_clusters, y_, f, shapes, 0.1) + reg_loss
            loss_runs = [loss_min] #[loss_min, loss_stol*100.0]
        if learn_type == 'sft2':
            loss_stol = get_loss_stol(n_clusters, y_, f, shapes, 0.1) + reg_loss
            loss_runs = [loss_stol]
        if learn_type == 'sft3':
            loss_stol = get_loss_stol(n_clusters, y_, f, shapes, 0.001) + reg_loss
            loss_runs = [loss_guess] #, loss_stol, loss_fun]
    return loss_runs 

def choose_trainable_variables(learn_type):
    train_var_lists = [tf.trainable_variables()]
    if learn_type == 'context': 
        train_var_lists = [[v for v in tf.trainable_variables() if "context" in v.name]]
    if learn_type == 'segment': 
        train_var_lists = [[v for v in tf.trainable_variables() if "function" in v.name]] # remember to change back to function!!!
        train_var_lists += [[v for v in tf.trainable_variables() if "shape" in v.name]]
    if learn_type == 'sft':
        train_var_lists = [[v for v in tf.trainable_variables() if "function" in v.name]] 
        #train_var_lists += [[v for v in tf.trainable_variables() if "shape" in v.name]]
    if learn_type == 'sft2':
        train_var_lists = [[v for v in tf.trainable_variables() if "shape" in v.name]]
    if learn_type == 'sft3':
        train_var_lists = [[v for v in tf.trainable_variables() if "transition" in v.name]]
        #train_var_lists += [[v for v in tf.trainable_variables() if "shape" in v.name]] 
        #train_var_lists += [[v for v in tf.trainable_variables() if "function" in v.name]]# [[v for v in tf.trainable_variables() if "shape" in v.name]]
    return train_var_lists

def get_loss_stol(n_clusters, y_, f, shapes, tol):
    tol_conditions_untiled = marked_within_tol(y_, f, tol)
    tol_conditions = tf.tile(tf.reshape(tol_conditions_untiled,[n_clusters,1,tf.size(y_)]),[1,3,1])
    #assigned_conditions_untiled = tf.logical_not(tf.tile(tf.equal(tf.count_nonzero(tol_conditions_untiled,0,True),0),[n_clusters,1])) # True across all clusters if at least one cluster matches perfectly
    #assigned_conditions = tf.tile(tf.reshape(assigned_conditions_untiled,[n_clusters,1,tf.size(y_)]),[1,3,1])
    #loss_s_in_tol = tf.where(tf.logical_and(assigned_conditions,tol_conditions),tf.maximum(tf.abs(shapes),1.0)-1.0,tf.zeros_like(shapes))
    #loss_s_not_in_tol = tf.where(tf.logical_and(assigned_conditions,tf.logical_not(tol_conditions)),1.0-tf.minimum(tf.abs(shapes),1.0),tf.zeros_like(shapes))
    loss_s_in_tol = tf.where(tol_conditions,tf.maximum(tf.abs(shapes),1.0)-0.999,tf.zeros_like(shapes))
    loss_s_not_in_tol = tf.where(tf.logical_not(tol_conditions),1.001-tf.minimum(tf.abs(shapes),1.0),tf.zeros_like(shapes))
    #loss_s_in_tol = tf.where(tol_conditions,tf.abs(shapes)-1.0,tf.zeros_like(shapes))
    #loss_s_not_in_tol = tf.where(tf.logical_not(tol_conditions),1.0-tf.abs(shapes),tf.zeros_like(shapes))
    loss_stol = tf.reduce_mean(loss_s_in_tol + loss_s_not_in_tol) # * 0.01
    return loss_stol #tf.divide(tf.reduce_sum(loss_stol),tf.count_nonzero(loss_stol,dtype = tf.float32))*0.1

def marked_within_tol(y_, f, tol_val = 0.05):
    tol = tf.constant([tol_val],dtype=tf.float32)
    diff_ratio = tf.abs(y_-f) # tf.divide(tf.abs(y_-f),tf.abs(y_))
    tol_conditions = tf.less(diff_ratio, tol)
    return tol_conditions

def get_loss_ctol(n_clusters, y_, f, c):
    tol_conditions = marked_within_tol(y_, f) # True where f matches y_ perfectly, False otherwise
    assigned_conditions = tf.logical_not(tf.tile(tf.equal(tf.count_nonzero(tol_conditions,0,True),0),[n_clusters,1])) # True across all clusters if at least one cluster matches perfectly
    c_zero_condition = tf.where(tf.less(tf.constant([0.0],dtype=tf.float32),c),c,tf.zeros_like(c)) # True if c is not already less than zero
    c_one_condition = tf.where(tf.less(c,tf.constant([1.0],dtype=tf.float32)),c,tf.zeros_like(c)) # True if c is not already greater than one
    loss_in_tol = tf.where(tf.logical_and(tol_conditions,assigned_conditions), -c_one_condition, tf.zeros_like(c)) # make c bigger than 1
    loss_not_in_tol = tf.where(tf.logical_and(tf.logical_not(tol_conditions),assigned_conditions), c_zero_condition, tf.zeros_like(c)) # make c smaller than 0
    #loss_in_tol = tf.where(tol_conditions, tf.abs(c-1), tf.zeros_like(c))
    #loss_not_in_tol = tf.where(tol_conditions, tf.abs(c), tf.zeros_like(c))
    loss_ctol = loss_in_tol + loss_not_in_tol # * 0.01
    return tf.reduce_sum(loss_ctol)

def get_loss_centers(n_clusters, y_, f, c):
    loss_min = tf.reduce_min(tf.abs(y_-f),0)
    loss_min_tiled = tf.tile(tf.reshape(loss_min,[1,tf.size(loss_min)]) ,[n_clusters,1])
    argmin_conditions = tf.equal(loss_min_tiled, tf.abs(y_-f))
    #argmin_conditions = tf.tile(tf.reshape(argmin_conditions_untiled,[n_clusters,1,tf.size(loss_min)]),[1,3,1])
    shape_conditions = get_closest_shape_conditions(c, n_clusters)
    loss_in_shape = tf.where(tf.logical_and(argmin_conditions,tf.logical_not(shape_conditions)), tf.reduce_sum(tf.abs(c),1), tf.zeros_like(f))
    loss_not_in_shape = tf.where(tf.logical_and(tf.logical_not(argmin_conditions),shape_conditions), tf.reduce_sum(-tf.abs(c),1), tf.zeros_like(f))
    loss_shape = loss_in_shape + loss_not_in_shape # * 0.01
    return loss_shape*100.0

def get_loss_shape(n_clusters, y_, f, c):
    #tol_conditions_untiled = marked_within_tol(y_, f)
    #print(tol_conditions_untiled)
    #tol_conditions = tf.tile(tf.reshape(tol_conditions_untiled,[n_clusters+1,1,tf.size(loss_min)]),[1,3,1])[:-1]
    #loss_function = mark_function(tf.abs(y_-f), tol_conditions)
    #loss_in_shape = mark_in_shape(c, argmin_points)
    #loss_not_in_shape = mark_not_in_shape(c, tf.logical_not(argmin_points))
    #loss_shape = loss_in_shape + loss_not_in_shape
    #loss_not_in_shape = tf.where(tf.logical_and(tol_conditions,loss_min_points), 1.0/tf.abs(c), tf.zeros_like(c))
    #loss_in_shape = tf.where(tf.logical_and(argmin_points,tol_conditions), tf.abs(c), tf.zeros_like(c))
    loss_min = tf.reduce_min(tf.abs(y_-f),0)
    loss_min_tiled = tf.tile(tf.reshape(loss_min,[1,tf.size(loss_min)]) ,[n_clusters,1]) #or [n_clusters+1,1]
    argmin_conditions_untiled = tf.equal(loss_min_tiled, tf.abs(y_-f))#[:-1]
    argmin_conditions = tf.tile(tf.reshape(argmin_conditions_untiled,[n_clusters,1,tf.size(loss_min)]),[1,3,1])
    shape_conditions = get_shape_conditions_c3(c)
    loss_min_but_not_in_shape = tf.where(tf.logical_and(argmin_conditions,tf.logical_not(shape_conditions)), tf.abs(c)-1.00, tf.zeros_like(c))
    loss_not_min_but_in_shape = tf.where(tf.logical_and(tf.logical_not(argmin_conditions),shape_conditions), 1.00-tf.abs(c), tf.zeros_like(c))
    #loss_in_shape = tf.where(argmin_conditions, tf.abs(c)*tf.cast(tf.logical_not(shape_conditions),tf.float32), tf.zeros_like(c))
    #loss_not_in_shape = tf.where(tf.logical_not(argmin_conditions), 1.0-tf.abs(c)*tf.cast(shape_conditions,tf.float32), tf.zeros_like(c))
    reduced_shape_conditions = get_shape_conditions_reduced_c3(c) # 5 x ?
    unassigned_conditions = tf.tile(tf.equal(tf.count_nonzero(reduced_shape_conditions,0,True),0),[n_clusters,1]) # only return true if all n_clusters are false
    nearest_cluster = get_closest_shape_conditions(c, n_clusters) #  5 x ? TFFFF x ?
    assign_nearest_conditions = tf.tile(tf.reshape(tf.logical_and(unassigned_conditions,nearest_cluster),[n_clusters,1,tf.size(loss_min)]),[1,3,1]) # logic and tile it by 1,3,1
    loss_push_unassigned_in_nearest_shape_center = tf.where(tf.logical_and(assign_nearest_conditions,tf.logical_not(shape_conditions)), tf.abs(c)-1.00, tf.zeros_like(c))
    loss_shape = loss_min_but_not_in_shape + loss_not_min_but_in_shape + loss_push_unassigned_in_nearest_shape_center # * 0.01
    return loss_shape*1000.0

def get_loss_shape_c1(n_clusters, y_, f, c):
    loss_min = tf.reduce_min(tf.abs(y_-f),0)
    loss_min_tiled = tf.tile(tf.reshape(loss_min,[1,tf.size(loss_min)]) ,[n_clusters,1])
    argmin_conditions = tf.equal(loss_min_tiled, tf.abs(y_-f))
    #argmin_conditions = tf.reshape(argmin_conditions_unshaped,[n_clusters,1,tf.size(loss_min)])
    shape_conditions = get_shape_conditions(c)
    loss_in_shape = tf.where(tf.logical_and(argmin_conditions,tf.logical_not(shape_conditions)), tf.abs(c)-1.00, tf.zeros_like(c))
    loss_not_in_shape = tf.where(tf.logical_and(tf.logical_not(argmin_conditions),shape_conditions), 1.00-tf.abs(c), tf.zeros_like(c))
    loss_shape = loss_in_shape + loss_not_in_shape # * 0.01
    return loss_shape*10000000.0

def mark_function(state,marked_points):
    marked_state = tf.zeros_like(state)
    marked_state = tf.where(marked_points, state, marked_state)
    return marked_state

def mark_in_shape(p,marked_points):
    p_marked_points = tf.logical_or(tf.less(tf.constant([0.1],dtype=tf.float32),p),tf.less(p,tf.constant([-0.1],dtype=tf.float32)))
    total_marked_points = tf.logical_and(marked_points, p_marked_points)
    marked_state = tf.where(total_marked_points, tf.abs(p), tf.zeros_like(p))
    return marked_state

def mark_not_in_shape(p,unmarked_points):
    p_marked_points = tf.logical_and(tf.less(p,tf.constant([0.1],dtype=tf.float32)),tf.less(tf.constant([-0.1],dtype=tf.float32),p))
    total_marked_points = tf.logical_and(unmarked_points, p_marked_points)
    marked_state = tf.where(total_marked_points, 1.0/tf.abs(p), tf.zeros_like(p))
    return marked_state

def define_loss_deriv(y_, y, train_shape, train_loss_deriv = True): #mu, x, t
    loss_deriv, loss_deriv2 = (0, 0)
    if train_loss_deriv == True:
        reshaped_y = tf.reshape(y,train_shape)
        reshaped_y_ = tf.reshape(y_,train_shape)
        dydx, dydt, dydu = get_deriv(reshaped_y)
        dy_dx, dy_dt, dy_du = get_deriv(reshaped_y_)
        #dydx2, dydxdt, dydxdu = get_deriv(dydx)
        #dy_dx2, dy_dxdt, dy_dxdu = get_deriv(dy_dx)
        #dydtdx, dydt2, dydtdu = get_deriv(dydt)
        #dy_dtdx, dy_dt2, dy_dtdu = get_deriv(dy_dt)
        #dydudx, dydudt, dydu2 = get_deriv(dydu)
        #dy_dudx, dy_dudt, dy_du2 = get_deriv(dy_du)
        loss_deriv = tf.reduce_sum(tf.abs(dy_dx-dydx)) + tf.reduce_sum(tf.abs(dy_dt-dydt)) + tf.reduce_sum(tf.abs(dy_du-dydu))*10 
        #loss_deriv2 = tf.reduce_sum(tf.abs(dy_dx2-dydx2)) + tf.reduce_sum(tf.abs(dy_dt2-dydt2)) + tf.reduce_sum(tf.abs(dy_du2-dydu2))*10
    return loss_deriv #+ loss_deriv2

def get_deriv(tf_vec):
    dfdx = tf_vec[:,:,1:]-tf_vec[:,:,:-1] 
    dfdt = tf_vec[:,1:,:]-tf_vec[:,:-1,:] 
    dfdu = tf_vec[1:,:,:]-tf_vec[:-1,:,:] 
    return dfdx, dfdt, dfdu

def tf_flat_vars(input_vars):
    if input_vars != []:
        #vars = tf.concat([tf.contrib.layers.flatten(tf.expand_dims(v,0))[:,1:] for v in input_vars],1)
        vars = tf.concat([tf.contrib.layers.flatten(tf.expand_dims(v,0))[:,:] for v in input_vars],1)
        other_vars = tf.concat([tf.contrib.layers.flatten(tf.expand_dims(v,0))[:,:0] for v in input_vars],1)
    else:
        vars, other_vars = (0,0)
    return vars, other_vars

def define_regularization_loss(learn_type, mult = 0.01):
    train_var_reg, train_other_var_reg = tf_flat_vars(tf.trainable_variables())
    reg_loss = tf_get_norm(train_var_reg,'L1')*mult #+ tf_get_norm(train_other_var_reg,'L1')*mult*0.01
    if learn_type == 'context': 
        train_var_reg, train_other_var_reg = tf_flat_vars([v for v in tf.trainable_variables() if "weights_layer%s"%str(len(n_c_hidden)) in v.name and "context" in v.name])
        reg_loss = -tf_get_norm(train_var_reg,'L2')*10  #only for regularization   
    if learn_type == 'segment' or learn_type == 'init_f' or learn_type == 'init_c':
        train_var_reg, train_other_var_reg = tf_flat_vars([v for v in tf.trainable_variables() if "s0" not in v.name]) #only for regularization
        reg_loss =  tf_get_norm(train_var_reg,'L1')*mult*0.001
    if learn_type == 'sft' or learn_type == 'sft2' or learn_type == 'sft3' or learn_type == 'mlsp':
        train_var_reg, train_other_var_reg = tf_flat_vars([v for v in tf.trainable_variables() if "s0" not in v.name]) #only for regularization
        reg_loss = tf_get_norm(train_var_reg,'L1')*mult #+ tf_get_norm(train_other_var_reg,'L1')*mult*0.01
    return reg_loss

def get_savename(learn_type, n_hidden, norm_type, lr_decay_type, lr_start, lr_decay):
    if learn_type == 'init_f' or learn_type == 'init_c':
        savename = 'segment'
    if learn_type == 'sft2' or learn_type == 'sft3':
        savename = 'sft'
    else:
        savename = '%s'%learn_type
    for i in n_hidden:
        savename += '_'+str(i)
    savename += '_'+lr_decay_type + '_'+str(lr_start)+'_'+str(lr_decay)+'_'+norm_type+'.npy'
    return savename

def initialize_network(savename,learn_type, train_state, input, y, y_, f, c):
    print("Initializing network")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if os.path.isfile('weights_'+savename) == True:
        saved_weights_init = np.load('weights_'+savename)
        [sess.run(var.assign(saved_weights_init[i])) for i, var in enumerate(tf.trainable_variables())]
    elif learn_type == 'init_f' or learn_type == 'init_c' or learn_type == 'segment' or learn_type == 'sft':
        print('Setting network weights according to pre-initialization heuristic')
        initialize_segment_network(sess, train_state, input, y, y_, f, c)
    return sess

def initialize_segment_network(sess, train_state, input, y, y_, f, c):
    shape_weights = [v for v in tf.trainable_variables() if "weights_s0" in v.name and "shape" in v.name]
    shape_biases = [v for v in tf.trainable_variables() if "biases_s0" in v.name and "shape" in v.name]
    function_weights = [v for v in tf.trainable_variables() if "weights_s0" in v.name and "function" in v.name]
    function_biases = [v for v in tf.trainable_variables() if "biases_s0" in v.name and "function" in v.name]
    n_clusters = len(shape_weights)
    train_shape = train_state[3].shape
    batch_inputs, batch_ys = get_batch(1.0, train_state)
    subset = {input: batch_inputs, y_:batch_ys}
    stimuli, temporal_points, spatial_points, clips = train_state
    for n in range(n_clusters):    
        if n == 0:
            mi0, mi1, mi2 = [0,0,0]
        #if n == 1:
        #    mi0, mi1, mi2 = [1,3,246]
        #if n == 2:
        #    mi0, mi1, mi2 = [1,3,100]
        else:    
            error_flat = sess.run(tf.reduce_min(tf.abs(y_-f),0), feed_dict=subset)
            error = np.reshape(error_flat, train_shape)
            error_max = sorted(np.unique(np.ndarray.flatten(error[:-1,:-1,:-1])))
            max_index = np.where(np.max(error[:-1,:-1,:-1]) == error[:-1,:-1,:-1])
            mi0, mi1, mi2 = list(np.transpose(max_index)[0])        
        tiny_state = stimuli[:], temporal_points[mi1:mi1+2], spatial_points[mi2:mi2+2], clips[mi0:mi0+2, mi1:mi1+2, mi2:mi2+2]
        tiny_inputs, tiny_ys = get_batch(1.0, tiny_state)
        tiny_inputs_bias = np.vstack((tiny_inputs,np.ones_like(tiny_ys)))
        Wbf, _, _, _ = np.linalg.lstsq(tiny_inputs_bias.T, tiny_ys.T)
        Wf, bf = Wbf[:-1].T, Wbf[-1:].T
        sess.run(function_weights[n].assign(Wf))
        sess.run(function_biases[n].assign(bf))
        ##Only uncomment if c3 is back to 3
        Ws = np.zeros((3,3))
        bs = np.zeros((3,1))
        for i in range(3):
            Ws[i,i] = 0.5/(np.max(tiny_state[i]) - np.min(tiny_state[i])) # or 2.0
            bs[i] = 0.25 - 0.5*np.max(tiny_state[i])/(np.max(tiny_state[i]) - np.min(tiny_state[i])) # or 1.0, 2.0
        sess.run(shape_weights[n].assign(Ws))
        sess.run(shape_biases[n].assign(bs)) 

def get_random_batch(batch_ratio, batch_all_inputs, batch_all_ys):
    n_all = batch_all_ys.shape[1]
    batch_size = int(batch_ratio*n_all)
    random_inds = np.random.randint(0,n_all-1,batch_size)
    batch_inputs, batch_ys = batch_all_inputs[:,random_inds], batch_all_ys[:,random_inds]
    return batch_inputs, batch_ys

def train_network(network_vars, learn_vars, train_state, batch_ratio, num_epochs):
    num_epochs_print = num_epochs
    _, _, _, desired_slope = learn_vars
    sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
    total_error = np.zeros((num_epochs))
    num_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    losses = [0]
    print('Training network...')
    #sess.run([v for v in tf.global_variables() if "global_step" in v.name][0].assign(0))
    j_printout = 200
    slope = 100
    starttime = time()
    batch_all_inputs, batch_all_ys = get_batch(1.0, train_state)
    for j in range(0,num_epochs):
        if batch_ratio != 1.0 or j == 0:
            batch_inputs, batch_ys = get_random_batch(batch_ratio, batch_all_inputs, batch_all_ys)
            subset = {input: batch_inputs, y_:batch_ys}
        #if j == 0 or (j % 10 == 0 and batch_ratio !=1.0):
        #    batch_inputs, batch_ys = get_batch(batch_ratio, train_state)
        #    subset = {input: batch_inputs, y_:batch_ys}
        if j % j_printout == 0:
            losses = [sess.run(loss_run, feed_dict=subset) for loss_run in loss_runs]
            lr_run = sess.run(learning_rate, feed_dict=subset)
        total_error[j] = sess.run(loss_runs[0], feed_dict=subset)
        if j < 1:
            [sess.run(train_step_run, feed_dict=subset) for train_step_run in train_step_runs]
        else:
            [sess.run(train_step_run, feed_dict=subset) for train_step_run in train_step_runs]
        if j % j_printout == 0:
            jrange = 250 if j > 249 else j
            slope = np.abs(linregress(range(jrange), total_error[j-jrange:j]).slope) if j > 10 else 0.1
            if slope < desired_slope:
                print('Converged! Convergence slope (final) on iteration %s is %s out of %s'%(j, slope,desired_slope))
                num_epochs_print = j
                break
            t_s = int((num_epochs-j)*(time()-starttime)/j) if j > 0 else 200000
            m, s = divmod(t_s, 60)
            print("Losses for epoch", j, "out of", num_epochs, "are", losses, "with lr", lr_run, 'and cs',slope,'out of',desired_slope,'with %s minutes %s seconds'% (m,s),'remaining') 
    network_vars = sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate
    return network_vars, total_error, num_epochs_print

def descriptive_clusters(n_clusters,train_state,f_run,y__run):
    #y__run = sess.run(y_,train_dict)
    #f_run = sess.run(f,train_dict)
    stimuli_full_range = train_state[0].shape[0]
    argmin2 = np.less(np.abs(y__run-f_run), 0.01).reshape(n_clusters,stimuli_full_range,train_state[3].shape[1]*train_state[3].shape[2])
    mu_exp = np.sort(np.sum(argmin2,2))[:,::-1] #0.1 just to avoid divide by 0 errors?
    heuristic = [mu_exp[j,1]/mu_exp[j,0] for j in range(n_clusters)]
    kept_clusters = [i for i, x in enumerate(np.array(heuristic) > 0.25) if x]
    return heuristic, kept_clusters

def descriptive_clusters_old(n_clusters,train_state,f_run,y__run):
    stimuli_full_range = train_state[0].shape[0]
    argmin2 = np.argmin(np.abs(y__run-f_run),0).reshape(stimuli_full_range,train_state[3].shape[1]*train_state[3].shape[2])
    mu_exp = [[np.float(np.sum(argmin2[i] == j)) for i in range(stimuli_full_range)] for j in range(n_clusters)] # to determine explanatory power in each mu
    heuristic = [(np.min(mu_exp[j])*np.sum(mu_exp[j]))/(np.max(mu_exp[j])*np.sum(mu_exp)) for j in range(n_clusters)]
    return heuristic, [i for i, x in enumerate(np.array(heuristic) > 0.01/n_clusters) if x]

def determine_n_clusters(sess,n_clusters,train_state,f_run,y__run):
    heuristic, kept_clusters = descriptive_clusters(n_clusters,train_state,f_run,y__run)
    save_vars = sum([tf.get_collection('trainable_variables',"function%s"% str(n)) for n in kept_clusters],[])\
              + sum([tf.get_collection('trainable_variables',"context%s"% str(n)) for n in kept_clusters],[])
    n_clusters_network = len(kept_clusters)
    print('Training found %s clusters'%n_clusters_network)
    if n_clusters_network != n_clusters:
        print("Reducing number of clusters from %s to %s" % (n_clusters,n_clusters_network))
    return heuristic, n_clusters_network, save_vars

def determine_n_segments(sess,learn_type,n_clusters,train_state,f_run,y__run):
    heuristic, kept_clusters = descriptive_clusters(n_clusters,train_state,f_run,y__run)
    save_vars = []
    for n in kept_clusters:
        save_vars += tf.get_collection('trainable_variables',"shape%s"% str(n))\
                   + tf.get_collection('trainable_variables',"function%s"% str(n))
        #if learn_type == 'sft' or learn_type == 'sft2' or learn_type == 'sft3':
        #    save_vars += tf.get_collection('trainable_variables',"transition%s"% str(n))
    if learn_type == 'sft' or learn_type == 'sft2' or learn_type == 'sft3':
        save_vars += tf.get_collection('trainable_variables',"transition")
    n_clusters_network = len(kept_clusters)
    print('Training', learn_type, 'network found %s clusters'%n_clusters_network)
    if n_clusters_network != n_clusters:
        print("Reducing number of clusters from %s to %s" % (n_clusters,n_clusters_network))
    return heuristic, n_clusters_network, save_vars

def check_convergence(savename, network_vars, n_clusters, train_state, learn_type):
    sess, input, y_, f, c, y, loss_runs, train_step_runs, learning_rate = network_vars
    # Get training inputs for entire network for heuristic checking
    train_inputs, train_ys = get_batch(1.0, train_state)
    train_dict = {input: train_inputs, y_:train_ys}
    # Determine number of clusters to keep and save weights for only those clusters
    if learn_type == 'initialize' or learn_type == 'context':
        converged = False
        heuristic, n_clusters_network, save_vars = determine_n_clusters(sess,n_clusters,train_state,sess.run(f,train_dict),sess.run(y_,train_dict))
        # Logic for total convergence
        if learn_type == 'context':
            converged = True
        if n_clusters_network == n_clusters and learn_type == 'initialize':
            learn_type = 'context'
    elif learn_type == 'init_c' or learn_type == 'init_c' or learn_type == 'sft' or learn_type == 'sft2':
        converged = False
        heuristic, n_clusters_network, save_vars = determine_n_segments(sess,learn_type,n_clusters,train_state,sess.run(f,train_dict),sess.run(y_,train_dict))
        if learn_type == 'init_c':
            learn_type = 'segment'
        if learn_type == 'init_f':
            learn_type = 'init_c'
        if n_clusters_network == n_clusters and learn_type == 'sft3':
            converged = True
        if learn_type == 'sft2':
            learn_type = 'sft3'
        if learn_type == 'sft':
            learn_type = 'sft2'
    else:
        converged, n_clusters_network, save_vars = (True, n_clusters, tf.trainable_variables()) 
    np.save('weights_'+savename,sess.run([var for var in save_vars]))
    return train_dict, n_clusters_network, learn_type, converged