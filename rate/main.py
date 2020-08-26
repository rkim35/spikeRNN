#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: Oct. 11, 2019
# Email: rkim@salk.edu
# Description: main script for training continuous-variable rate RNN models 
# For more info, refer to 
# Kim R., Li Y., & Sejnowski TJ. Simple Framework for Constructing Functional Spiking 
# Recurrent Neural Networks. Preprint at BioRxiv 
# https://www.biorxiv.org/content/10.1101/579706v2 (2019).

import os, sys
import time
import scipy.io
import numpy as np
import tensorflow as tf
import argparse
import datetime

# Import utility functions
from utils import set_gpu
from utils import restricted_float
from utils import str2bool

# Import the continuous rate model
from model import FR_RNN_dale

# Import the tasks
from model import generate_input_stim_xor
from model import generate_target_continuous_xor

from model import generate_input_stim_mante
from model import generate_target_continuous_mante

from model import generate_input_stim_go_nogo
from model import generate_target_continuous_go_nogo

from model import construct_tf
from model import loss_op

# Parse input arguments
parser = argparse.ArgumentParser(description='Training rate RNNs')
parser.add_argument('--gpu', required=False,
        default='0', help="Which gpu to use")
parser.add_argument("--gpu_frac", required=False,
        type=restricted_float, default=0.4,
        help="Fraction of GPU mem to use")
parser.add_argument("--n_trials", required=True,
        type=int, default=200, help="Number of epochs")
parser.add_argument("--mode", required=True,
        type=str, default='Train', help="Train or Eval")
parser.add_argument("--output_dir", required=True,
        type=str, help="Model output path")
parser.add_argument("--N", required=True,
        type=int, help="Number of neurons")
parser.add_argument("--gain", required=False,
        type=float, default = 1.5, help="Gain for the connectivity weight initialization")
parser.add_argument("--P_inh", required=False,
        type=restricted_float, default = 0.20,
        help="Proportion of inhibitory neurons")
parser.add_argument("--P_rec", required=False,
        type=restricted_float, default = 0.20,
        help="Connectivity probability")
parser.add_argument("--som_N", required=True,
        type=int, default = 0, help="Number of SST neurons")
parser.add_argument("--task", required=True,
        type=str, help="Task (XOR, sine, etc...)")
parser.add_argument("--act", required=True,
        type=str, default='sigmoid', help="Activation function (sigmoid, clipped_relu)")
parser.add_argument("--loss_fn", required=True,
        type=str, default='l2', help="Loss function (either L1 or L2)")
parser.add_argument("--apply_dale", required=True,
        type=str2bool, default='True', help="Apply Dale's principle?")
parser.add_argument("--decay_taus", required=True,
        nargs='+', type=float,
        help="Synaptic decay time-constants (in time-steps). If only one number is given, then all\
        time-constants set to that value (i.e. not trainable). Otherwise specify two numbers (min, max).")
args = parser.parse_args()

# Set up the output dir where the output model will be saved
out_dir = os.path.join(args.output_dir, 'models', args.task.lower())
if args.apply_dale == False:
    out_dir = os.path.join(out_dir, 'NoDale')
if len(args.decay_taus) > 1:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Taus_' + str(args.decay_taus[0]) + '_' + str(args.decay_taus[1]))
else:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Tau_' + str(args.decay_taus[0]))

if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

# Number of units/neurons
N = args.N
som_N = args.som_N; # number of SST neurons 

# Define task-specific parameters
# NOTE: Each time step is 5 ms
if args.task.lower() == 'go-nogo':
    # GO-NoGo task
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'xor':
    # XOR task 
    settings = {
            'T': 300, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 50, # input stim duration (in steps)
            'delay': 10, # delay b/w the two stimuli (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'mante':
    # Sensory integration task
    settings = {
            'T': 500, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 200, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }

'''
Initialize the input and output weight matrices
'''
# Go-Nogo task
if args.task.lower() == 'go-nogo':
    w_in = np.float32(np.random.randn(N, 1))
    w_out = np.float32(np.random.randn(1, N)/100)

# XOR task
elif args.task.lower() == 'xor':
    w_in = np.float32(np.random.randn(N, 2))
    w_out = np.float32(np.random.randn(1, N)/100)

# Sensory integration task
elif args.task.lower() == 'mante':
    w_in = np.float32(np.random.randn(N, 4))
    w_out = np.float32(np.random.randn(1, N)/100)

'''
Initialize the continuous rate model
'''
P_inh = args.P_inh # inhibitory neuron proportion
P_rec = args.P_rec # initial connectivity probability (i.e. sparsity degree)
print('P_rec set to ' + str(P_rec))

w_dist = 'gaus' # recurrent weight distribution (Gaussian or Gamma)
net = FR_RNN_dale(N, P_inh, P_rec, w_in, som_N, w_dist, args.gain, args.apply_dale, w_out)
print('Intialized the network...')


'''
Define the training parameters (learning rate, training termination criteria, etc...)
'''
training_params = {
        'learning_rate': 0.01, # learning rate
        'loss_threshold': 7, # loss threshold (when to stop training)
        'eval_freq': 100, # how often to evaluate task perf
        'eval_tr': 100, # number of trials for eval
        'eval_amp_threh': 0.7, # amplitude threshold during response window
        'activation': args.act.lower(), # activation function
        'loss_fn': args.loss_fn.lower(), # loss function ('L1' or 'L2')
        'P_rec': 0.20
        }


'''
Construct the TF graph for training
'''
if args.mode.lower() == 'train':
    input_node, z, x, r, o, w, w_in, m, som_m, w_out, b_out, taus\
            = construct_tf(net, settings, training_params)
    print('Constructed the TF graph...')

    # Loss function and optimizer
    loss, training_op = loss_op(o, z, training_params)


'''
Start the TF session and train the network
'''
sess = tf.Session(config=tf.ConfigProto(gpu_options=set_gpu(args.gpu, args.gpu_frac)))
init = tf.global_variables_initializer()

if args.mode.lower() == 'train':
    with tf.Session() as sess:
        print('Training started...')
        init.run()
        training_success = False

        if args.task.lower() == 'go-nogo':
            # Go-NoGo task
            u, label = generate_input_stim_go_nogo(settings)
            target = generate_target_continuous_go_nogo(settings, label)
            x0, r0, w0, w_in0, taus_gaus0 = \
                    sess.run([x, r, w, w_in, taus], feed_dict={input_node: u, z: target})

        elif args.task.lower() == 'xor':
            # XOR task
            u, label = generate_input_stim_xor(settings)
            target = generate_target_continuous_xor(settings, label)
            x0, r0, w0, w_in0, taus_gaus0 = \
                    sess.run([x, r, w, w_in, taus], feed_dict={input_node: u, z: target})

        elif args.task.lower() == 'mante':
            # Sensory integration task
            u, label = generate_input_stim_mante(settings)
            target = generate_target_continuous_mante(settings, label)
            x0, r0, w0, w_in0, taus_gaus0 = \
                    sess.run([x, r, w, w_in, taus], feed_dict={input_node: u, z: target})

        # For storing all the loss vals
        losses = np.zeros((args.n_trials,))

        for tr in range(args.n_trials):
            start_time = time.time()

            # Generate a task-specific input signal
            if args.task.lower() == 'go-nogo':
                u, label = generate_input_stim_go_nogo(settings)
                target = generate_target_continuous_go_nogo(settings, label)
            elif args.task.lower() == 'xor':
                u, label = generate_input_stim_xor(settings)
                target = generate_target_continuous_xor(settings, label)
            elif args.task.lower() == 'mante':
                u, label = generate_input_stim_mante(settings)
                target = generate_target_continuous_mante(settings, label)

            print("Trial " + str(tr) + ': ' + str(label))

            # Train using backprop
            _, t_loss, t_w, t_o, t_w_out, t_x, t_r, t_m, t_som_m, t_w_in, t_b_out, t_taus_gaus = \
                    sess.run([training_op, loss, w, o, w_out, x, r, m, som_m, w_in, b_out, taus],
                    feed_dict={input_node: u, z: target})

            print('Loss: ', t_loss)
            losses[tr] = t_loss

            '''
            Evaluate the model and determine if the training termination criteria are met
            '''
            # Go-NoGo task
            if args.task.lower() == 'go-nogo':
                resp_onset = settings['stim_on'] + settings['stim_dur']
                if (tr-1)%training_params['eval_freq'] == 0:
                    eval_perf = np.zeros((1, training_params['eval_tr']))
                    eval_losses = np.zeros((1, training_params['eval_tr']))
                    eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                    eval_labels = np.zeros((training_params['eval_tr'], ))
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_label = generate_input_stim_go_nogo(settings)
                        eval_target = generate_target_continuous_go_nogo(settings, eval_label)
                        eval_o, eval_l = sess.run([o, loss], feed_dict = \
                                {input_node: eval_u, z: eval_target})
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o
                        eval_labels[ii, ] = eval_label
                        if eval_label == 1:
                            if np.max(eval_o[resp_onset:]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.max(np.abs(eval_o[resp_onset:])) < 0.3:
                                eval_perf[0, ii] = 1

                    eval_perf_mean = np.nanmean(eval_perf, 1)
                    eval_loss_mean = np.nanmean(eval_losses, 1)
                    print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                    if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95 and tr > 1500:
                        # For this task, the minimum number of trials required is set to 1500 to 
                        # ensure that the trained rate model is stable.
                        training_success = True
                        break

            # XOR task
            elif args.task.lower() == 'xor':
                if (tr-1)%training_params['eval_freq'] == 0:
                    eval_perf = np.zeros((1, training_params['eval_tr']))
                    eval_losses = np.zeros((1, training_params['eval_tr']))
                    eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                    eval_labels = []
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_label = generate_input_stim_xor(settings)
                        eval_target = generate_target_continuous_xor(settings, eval_label)
                        eval_o, eval_l = sess.run([o, loss], feed_dict = \
                                {input_node: eval_u, z: eval_target})
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o
                        eval_labels.append(eval_label)
                        if eval_label == 'same':
                            if np.max(eval_o[200:]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.min(eval_o[200:]) < -training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1

                    eval_perf_mean = np.nanmean(eval_perf, 1)
                    eval_loss_mean = np.nanmean(eval_losses, 1)
                    print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                    if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95:
                        training_success = True
                        break

            # Sensory integration task
            elif args.task.lower() == 'mante':
                if (tr-1)%training_params['eval_freq'] == 0:
                    eval_perf = np.zeros((1, training_params['eval_tr']))
                    eval_losses = np.zeros((1, training_params['eval_tr']))
                    eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                    eval_labels = np.zeros((training_params['eval_tr'], ))
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_label = generate_input_stim_mante(settings)
                        eval_target = generate_target_continuous_mante(settings, eval_label)
                        eval_o, eval_l = sess.run([o, loss], feed_dict = \
                                {input_node: eval_u, z: eval_target})
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o
                        eval_labels[ii, ] = eval_label
                        if eval_label == 1:
                            if np.max(eval_o[-200:]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.min(eval_o[-200:]) < -training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1

                    eval_perf_mean = np.nanmean(eval_perf, 1)
                    eval_loss_mean = np.nanmean(eval_losses, 1)
                    print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                    if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95:
                        training_success = True
                        break

        elapsed_time = time.time() - start_time
        # print(elapsed_time)

        # Save the trained params in a .mat file
        var = {}
        var['x0'] = x0
        var['r0'] = r0
        var['w0'] = w0
        var['taus_gaus0'] = taus_gaus0
        var['w_in0'] = w_in0
        var['u'] = u
        var['o'] = t_o
        var['w'] = t_w
        var['x'] = t_x
        var['target'] = target
        var['w_out'] = t_w_out
        var['r'] = t_r
        var['m'] = t_m
        var['som_m'] = t_som_m
        var['N'] = N
        var['exc'] = net.exc
        var['inh'] = net.inh
        var['w_in'] = t_w_in
        var['b_out'] = t_b_out
        var['som_N'] = som_N
        var['losses'] = losses
        var['taus'] = settings['taus']
        var['eval_perf_mean'] = eval_perf_mean
        var['eval_loss_mean'] = eval_loss_mean
        var['eval_os'] = eval_os
        var['eval_labels'] = eval_labels
        var['taus_gaus'] = t_taus_gaus
        var['tr'] = tr
        var['activation'] = training_params['activation']
        fname_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        if len(settings['taus']) > 1:
            fname = 'Task_{}_N_{}_Taus_{}_{}_Act_{}_{}.mat'.format(args.task.lower(), N, settings['taus'][0], 
                    settings['taus'][1], training_params['activation'], fname_time)
        elif len(settings['taus']) == 1:
            fname = 'Task_{}_N_{}_Tau_{}_Act_{}_{}.mat'.format(args.task.lower(), N, settings['taus'][0], 
                    training_params['activation'], fname_time)
        scipy.io.savemat(os.path.join(out_dir, fname), var)


