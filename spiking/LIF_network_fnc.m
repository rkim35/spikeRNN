% Name: Robert Kim
% Date: October 11, 2019
% Email: rkim@salk.edu
% LIF_network_RK.m
% Description: Function to perform the one-to-one mapping
% from a trained rate RNN to a spiking RNN (leaky integrate-and-fire).
% NOTE: LIF network implementation modified from LIFFORCESINE.m from NicolaClopath2017 
% (https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=190565&file=/NicolaClopath2017/)

function [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(model_path,...
scaling_factor, u, stims, downsample, use_initial_weights)
% FUNCTION LIF_network_fnc
% INPUT
%   - model_path: trained model full path (directory + filename)
%   - scaling_factor: scaling factor for transferring weights from rate to spk
%   - u: input stimulus to be used
%   - stims: struct for artificial stimulations (to model optogenetic stim)
%       - mode: "none", "exc" (depolarizing), or "inh" (hyperpolarizing)
%       - dur: [stim_onset stim_offset]
%       - units: vector containing unit indices to be stimulated
%   - downsample: downsample factor (1 => no downsampling, 2 => every other sample, etc...)
%                 While downsample > 1 can speed up the conversion, the LIF network
%                 might not be as robust as the one without downsampling
%   - use_initial_weights: whether to use w0 (random initial weights). This is mainly used
%                          for testing.
%
% OUTPUT
%   - W: recurrent connectivity matrix scaled by the scaling factor (N x N)
%   - REC: membrane voltage from all the units (N x t)
%   - spk: binary matrix indicating spikes (N x t)
%   - rs: firing rates from all the units (N x t)
%   - all_fr: average firing rates from all the units (N x 1)
%   - out: network output (1 x t)
%   - params: struct containing sampling rate info

%------------------------------------------------------
% Extract the number of units and the connectivity
% matrix from the trained continuous rate model
%------------------------------------------------------
load(model_path, 'w_in', 'w', 'w0', 'N', 'm', 'som_m', 'w_out', ...
'inh', 'exc', 'taus_gaus0', 'taus_gaus', 'taus');

% Number of neurons and the trained connectivity weight
% matrix (extracted from the trained continuous rate model)
N = double(N); 

% Shuffle nonzero weights
if use_initial_weights == true
  w = w0*m.*som_m;
else
  w = w*m.*som_m;
end

% Scale the connectivity weights by the optimal scaling factor 
W = w/scaling_factor;

% Inhibitory and excitatory neurons
inh_ind = find(inh);
exc_ind = find(exc);

% Input stimulus
u = u(:, 1:downsample:end);
ext_stim = w_in*u;

%------------------------------------------------------
% LIF network parameters
%------------------------------------------------------
dt = 0.00005*downsample;    % sampling rate
T = (size(u, 2)-1)*dt*100;  % trial duration (in sec)
nt = round(T/dt);           % total number of points in a trial
tref = 0.002;               % refractory time constant (in sec)
tm = 0.010;                 % membrane time constant (in sec)
vreset = -65;               % voltage reset (in mV)
vpeak = -40;                % voltage peak (in mV) for linear LIF
%vpeak = 30;                % voltage peak (in mV) for quadratic LIF
%rng(1);

% Synaptic decay time constants (in sec) for the double-exponential
% synpatic filter
% tr: rising time constant
% td: decay time constants
% td0: initial decay time constants (before optimization)
if length(taus) > 1
    td = (1./(1+exp(-taus_gaus))*(taus(2) - taus(1))+taus(1))*5/1000; 
    td0 = (1./(1+exp(-taus_gaus0))*(taus(2) - taus(1))+taus(1))*5/1000;
    tr = 0.002;
else
    td = taus*5/1000; 
    td0 = td;
    tr = 0.002;
end

% Synaptic parameters
IPSC = zeros(N,1);      % post synaptic current storage variable
h = zeros(N,1);         % storage variable for filtered firing rates
r = zeros(N,1);         % second storage variable for filtered rates
hr = zeros(N,1);        % third variable for filtered rates
JD = 0*IPSC;            % storage variable required for each spike time
tspike = zeros(4*nt,2); % storage variable for spike times
ns = 0;                 % number of spikes, counts during simulation

v = vreset + rand(N,1)*(30-vreset); % initialize voltage with random distribtuions
v_ = v;   % v_ is the voltage at previous time steps
v0 = v;   % store the initial voltage values

% Record REC (membrane voltage), Is (input currents), 
% spk (spike raster), rs (firing rates) from all the units
REC = zeros(nt,N);  % membrane voltage (in mV) values
Is = zeros(N, nt);  % input currents from the ext_stim
IPSCs = zeros(N, nt); % IPSC over time
spk = zeros(N, nt); % spikes
rs = zeros(N, nt);  % firing rates
hs = zeros(N, nt); % filtered firing rates

% used to set the refractory times
tlast = zeros(N,1); 

% Constant bias current to be added to ensure the baseline membrane voltage
% is around the rheobase
BIAS = vpeak; % for linear LIF
%BIAS = 0; % for quadratic LIF

%------------------------------------------------------
% Start the simulation
%------------------------------------------------------
for i = 1:nt
    IPSCs(:, i) = IPSC; % record the IPSC over time (comment out if not used to save time)

    I = IPSC + BIAS; % synaptic current

    % Apply external input stim if there is any
    I = I + ext_stim(:, round(i/100)+1);
    Is(:, i) = ext_stim(:, round(i/100)+1);

    % LIF voltage equation with refractory period
    dv = (dt*i>tlast + tref).*(-v+I)/tm; % linear LIF
    %dv = (dt*i>tlast + tref).*(v.^2+I)/tm; % quadratic LIF
    v = v + dt*(dv) + randn(N, 1)/10;

    % Artificial stimulation/inhibition
    if strcmpi(stims.mode, 'exc')
      if i >= stims.dur(1) & i < stims.dur(2)
        if rand < 0.50
          v(stims.units) = v(stims.units) + 0.5;
        end
      end
    elseif strcmpi(stims.mode, 'inh')
      if i >= stims.dur(1) & i < stims.dur(2)
        if rand < 0.50
          v(stims.units) = v(stims.units) - 0.5;
        end
      end
    end

    % find the neurons that have fired
    index = find(v>=vpeak);  

    % store spike times, and get the weight matrix column sum of spikers
    if length(index)>0
      JD = sum(W(:,index),2); %compute the increase in current due to spiking
      tspike(ns+1:ns+length(index),:) = [index, [0*index+dt*i]];
      ns = ns + length(index);  % total number of psikes so far
    end

    % used to set the refractory period of LIF neurons
    tlast = tlast + (dt*i -tlast).*(v>=vpeak);      

    % if the rise time is 0, then use the single synaptic filter,
    % otherwise (i.e. rise time is positive) use the double filter
    if tr == 0
        IPSC = IPSC.*exp(-dt./td)+JD*(length(index)>0)./(td);
        r = r.*exp(-dt./td) + (v>=vpeak)./td;
        rs(:, i) = r;
    else
        IPSC = IPSC.*exp(-dt./td) + h*dt;
        h = h*exp(-dt/tr) + JD*(length(index)>0)./(tr*td);  %Integrate the current
        hs(:, i) = h;

        r = r.*exp(-dt./td) + hr*dt;

        hr = hr*exp(-dt/tr) + (v>=vpeak)./(tr.*td);

        rs(:, i) = r;
    end

    % record the spikes
    spk(:, i) = v >= vpeak;
    
    v = v + (30 - v).*(v>=vpeak);

    % record the membrane voltage tracings from all the units
    REC(i,:) = v(1:N); 

    %reset with spike time interpolant implemented.
    v = v + (vreset - v).*(v>=vpeak); 
end

time = 1:1:nt;

% Plot the population response
out = w_out/scaling_factor*rs;

% Compute average firing rate for each population (excitatory/inhibitory)
inh_fr = zeros(size(inh_ind));
for i = 1:length(inh_ind)
  inh_fr(i) = length(find(spk(inh_ind(i), :)>0))/T;
end

exc_fr = zeros(size(exc_ind));
for i = 1:length(exc_ind)
  exc_fr(i) = length(find(spk(exc_ind(i), :)>0))/T;
end

all_fr = zeros(size(exc));
for i = 1:N
  all_fr(i) = length(find(spk(i, 10:end)>0))/T;
end

REC = REC';

% Some params
params = {};
params.dt =  dt;
params.T = T;
params.nt = nt;
params.w_out = w_out;
params.td = td;
params.td0 = td0;
params.IPSCs = IPSCs;



