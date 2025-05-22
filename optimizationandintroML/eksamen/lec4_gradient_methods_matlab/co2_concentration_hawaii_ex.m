clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONFIG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% import data (The complete data set)
data = import_co2_concentration('co2_weekly_mlo.txt');
N = size(data,1); % Number of data points

% Fill missing values
int_thr = 1;
data.int_idx = data.co2_ppm<int_thr;
data.co2_ppm(data.int_idx) = nan;
data.co2_ppm = fillmissing(data.co2_ppm,'movmedian',10);
data.dt = day_counter2datetime(data);

% Split data
Q = floor(N/2);

% Data used for the fit
data_fit = data(1:Q,:);

% Data used for validation
data_val = data(Q+1:N,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILL IN CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(data_val.year_dec, data_val.co2_ppm)
grid on
X = fitlm(data_val.year_dec,data_val.co2_ppm);
figure;
plotResiduals(X, 'fitted'); % Residuals vs Fitted
title('Residuals vs Fitted Values');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions (Must be in end of file)
function dt = day_counter2datetime(data)
    dt =datetime(data.year, data.month, data.day);
end