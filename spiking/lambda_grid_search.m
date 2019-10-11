% Name: Robert Kim
% Date: October 11, 2019
% Email: rkim@salk.edu
% lambda_grid_search.m
% Description: Script to perform grid search to determine
% the optimal scaling factor (lambda) for one-to-one mapping
% from a trained rate RNN to a LIF RNN
% NOTE
%   - The script utilizes the MATLAB Parallel Computing Toolbox
%   to speed up the script. This is not required, but it will
%   significantly speed up the script.
%   - Downsampling is turned off (i.e. set to 1). This can be
%   turned on (i.e. setting to a positive integer > 1) to speed up
%   the script, but the resulting LIF network might not be as robust
%   as the one constructed without downsampling.
%   - The script will perform the grid search on all the trained models
%   specified in "model_dir". It is set up in a way that allows
%   you to run multiple instances of the script. For example, if you have
%   access to 10 MATLAB licenses and 10 trained RNNs, then you can run
%   this script 10 times concurrently (using 10 separate MATLAB licenses).
%   - For each model in "model_dir", the script computes the task performance
%   for each scaling factor value ("scaling_factors"). The factor value with
%   the best performance is the optimal scaling factor ("opt_scaling_factor").
%   This value is appended to the model mat file.

clear; clc;
addpath('/cnl/chaos/ROBERT/spiking_working_memory/code/RK_TF_RNN');

% Directory containing all the trained rate RNN model .mat files
model_dir = '../models/go-nogo/P_rec_0.2_Taus_4.0_20.0';
mat_files = dir(fullfile(model_dir, '*.mat'));

% Whether to use the initial random connectivity weights
% This should be set to false unless you want to compare
% the effects of pre-trained vs post-trained weights
use_initial_weights = false; 

% Number of trials to use to evaluate the LIF RNN
n_trials = 100;

% Scaling factor values to try for grid search
% The more values it has, the longer the search
scaling_factors = [20:5:75];

% Grid search
for i = 1:length(mat_files)
  curr_fname = mat_files(i).name;
  curr_full = fullfile(mat_files(i).folder, curr_fname);
  disp(['Analyzing ' curr_fname]);

  % Get the task name
  if ~isempty(findstr(curr_full, 'go-nogo'))
    task_name = 'go-nogo';
  elseif ~isempty(findstr(curr_full, 'mante'))
    task_name = 'mante';
  elseif ~isempty(findstr(curr_full, 'xor'))
    task_name = 'xor';
  end

  % Load the model
  load(curr_full);

  % Skip if the file was run before
  if exist('opt_scaling_factor')
    clearvars -except model_dir mat_files n_trials scaling_factors use_initial_weights
    continue;
  else
    opt_scaling_factor = NaN;
    save(curr_full, 'opt_scaling_factor', '-append');
  end

  % Go-NoGo task
  if strcmpi(task_name, 'go-nogo')
    down_sample = 1;
    all_perfs = zeros(length(scaling_factors), 1);

    for k = 1:length(scaling_factors)
      outs = zeros(n_trials, 20000);
      trials = zeros(n_trials, 1);
      perfs = zeros(n_trials, 1);

      scaling_factor = scaling_factors(k);
      disp(scaling_factor)

      parfor j = 1:n_trials
        u = zeros(1, 201);
        if rand >= 0.50
          u(51:75) = 1.0;
          trials(j) = 1;
        end
        stims = struct();
        stims.mode = 'none';
        [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
        outs(j, :) = out;
        if max(out(10000:end)) > 0.7 & trials(j) == 1
          perfs(j) = 1;
        elseif max(out(10000:end)) < 0.3 & trials(j) == 0
          perfs(j) = 1;
        end
      end
      all_perfs(k) = mean(perfs);
    end
    [v, ind] = max(all_perfs);
    [v, scaling_factors(ind)]

    % Save the optimal scaling factor
    opt_scaling_factor = scaling_factors(ind);
    save(curr_full, 'opt_scaling_factor', 'all_perfs', 'scaling_factors', '-append');
    clear opt_scaling_factor;

  % Sensory integration task
  elseif strcmpi(task_name, 'mante')
    down_sample = 1;
    all_perfs = zeros(length(scaling_factors), 1);

    for k = 1:length(scaling_factors)
      outs = zeros(n_trials, 50000);
      trials = zeros(n_trials, 1);
      perfs = zeros(n_trials, 1);

      scaling_factor = scaling_factors(k);
      disp(scaling_factor)
      parfor j = 1:n_trials
        u = zeros(4, 501);
        u_lab = zeros(2, 1);

        % Stim 1
        if rand >= 0.50
          u(1, 51:250) = randn(1, 200) + 0.5;
          u_lab(1, 1) = 1;
        else
          u(1, 51:250) = randn(1, 200) - 0.5;
          u_lab(1, 1) = -1;
        end

        % Stim 2
        if rand >= 0.50
          u(2, 51:250) = randn(1, 200) + 0.5;
          u_lab(2, 1) = 1;
        else
          u(2, 51:250) = randn(1, 200) - 0.5;
          u_lab(2, 1) = -1;
        end

        % Context
        if rand >= 0.50
          u(3, :) = 1;
          if u_lab(1, 1) == 1
            label = 1;
          elseif u_lab(1, 1) == -1
            label = -1;
          end
        else
          u(4, :) = 1;
          if u_lab(2, 1) == 1
            label = 1;
          elseif u_lab(2, 1) == -1
            label = -1;
          end
        end
        trials(j) = label;

        stims = struct();
        stims.mode = 'none';
        [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, scaling_factor,...
            u, stims, down_sample, use_initial_weights);
        outs(j, :) = out;
        if label == 1
          if max(out(26000:end)) > 0.7
            perfs(j) = 1;
          end
        elseif label == -1
          if min(out(26000:end)) < -0.7
            perfs(j) = 1;
          end
        end
      end % parfor end
      all_perfs(k) = mean(perfs);

    end % scaling end
    [v, ind] = max(all_perfs);
    [v, scaling_factors(ind)]

    % Save the optimal scaling factor
    opt_scaling_factor = scaling_factors(ind);
    save(curr_full, 'opt_scaling_factor', 'all_perfs', 'scaling_factors', '-append');
    clear opt_scaling_factor;

  % XOR task
  elseif strcmpi(task_name, 'xor')
    down_sample = 1;
    all_perfs = zeros(length(scaling_factors), 1);

    for k = 1:length(scaling_factors)
      outs = zeros(n_trials, 30000);
      trials = zeros(n_trials, 1);
      perfs = zeros(n_trials, 1);

      scaling_factor = scaling_factors(k);
      disp(scaling_factor)
      parfor j = 1:n_trials
        u = zeros(2, 301);
        u_lab = zeros(1, 2);

        % Stim 1
        if rand >= 0.50
          u(1, 51:100) = 1;
          u_lab(1) = 1;
        else
          u(1, 51:100) = -1;
          u_lab(1) = -1;
        end

        % Stim 2
        if rand >= 0.50
          u(2, 111:160) = 1;
          u_lab(2) = 1;
        else
          u(2, 111:160) = -1;
          u_lab(2) = -1;
        end
        label = prod(u_lab);
        trials(j) = label;

        stims = struct();
        stims.mode = 'none';
        [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, scaling_factor,...
            u, stims, down_sample, use_initial_weights);
        outs(j, :) = out;
        if label == 1
          if max(out(20000:end)) > 0.7
            perfs(j) = 1;
          end
        elseif label == -1
          if min(out(20000:end)) < -0.7
            perfs(j) = 1;
          end
        end
      end % parfor end
      all_perfs(k) = mean(perfs);

    end % scaling end
    [v, ind] = max(all_perfs);
    [v, scaling_factors(ind)]

    % Save the optimal scaling factor
    opt_scaling_factor = scaling_factors(ind);
    save(curr_full, 'opt_scaling_factor', 'all_perfs', 'scaling_factors', '-append');
    clearvars -except model_dir mat_files n_trials scaling_factors use_initial_weights
  end
end


