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
degree = 3;
p = polyfit(data_fit.year, data_fit.co2_ppm, degree);
fit_pred = polyval(p, data_fit.year);
val_pred = polyval(p, data_val.year);
hold on
plot(data_val.year,val_pred, 'r-', 'LineWidth', 2)  % Fitted curve
hold on
plot(data.year,data.co2_ppm)
legend('extra','original');
grid on
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions (Must be in end of file)
function dt = day_counter2datetime(data)
    dt =datetime(data.year, data.month, data.day);
end