%% Part (b): 100 Basis Functions with Regularization
clear; close all; clc;
data = import_co2_concentration('co2_weekly_mlo.txt');
time = data.year;
co2 = data.co2_ppm;

time_normalized = (time - min(time)) / (max(time) - min(time));
split_idx = floor(length(time)/2);
time_fit = time_normalized(1:split_idx);
co2_fit = co2(1:split_idx);
time_val = time_normalized(split_idx+1:end);
co2_val = co2(split_idx+1:end);

%% Create 100 basis functions
Phi_100_fit = create_100_basis_functions(time_fit);
Phi_100_val = create_100_basis_functions(time_val);

% For future extrapolation (2030)
future_years = (2021:2030)';
future_time = (future_years - min(time)) / (max(time) - min(time));
Phi_100_future = create_100_basis_functions(future_time);

%% Define lambda range
lambda_values = logspace(-4, 3, 50); % Wider range for more basis functions

%% Ridge Regression with 100 basis functions
ridge_val_errors_100 = zeros(size(lambda_values));
ridge_coeffs_100 = zeros(size(Phi_100_fit,2), length(lambda_values));

for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    w = (Phi_100_fit'*Phi_100_fit + lambda*eye(size(Phi_100_fit,2))) \ (Phi_100_fit'*co2_fit);
    ridge_coeffs_100(:,i) = w;
    pred = Phi_100_val * w;
    ridge_val_errors_100(i) = sqrt(mean((pred - co2_val).^2));
end

[best_ridge_error_100, best_ridge_idx_100] = min(ridge_val_errors_100);
best_ridge_lambda_100 = lambda_values(best_ridge_idx_100);

%% Lasso Regression with 100 basis functions
[lasso_coeffs_100, fit_info_100] = lasso(Phi_100_fit(:,2:end), co2_fit, ...
    'Lambda', lambda_values, 'Standardize', false);

lasso_val_errors_100 = zeros(size(lambda_values));
for i = 1:length(lambda_values)
    pred = fit_info_100.Intercept(i) + Phi_100_val(:,2:end)*lasso_coeffs_100(:,i);
    lasso_val_errors_100(i) = sqrt(mean((pred - co2_val).^2));
end

[best_lasso_error_100, best_lasso_idx_100] = min(lasso_val_errors_100);
best_lasso_lambda_100 = lambda_values(best_lasso_idx_100);

%% Compare with unregularized solution (will likely be terrible)
w_unreg_100 = Phi_100_fit \ co2_fit;
unreg_val_error_100 = sqrt(mean((Phi_100_val*w_unreg_100 - co2_val).^2));

%% Plot results
figure;

% 1. Coefficient paths
subplot(2,2,1);
semilogx(lambda_values, ridge_coeffs_100', 'LineWidth', 0.5);
xlabel('Lambda'); ylabel('Coefficient value');
title('Ridge: 100 Basis Functions');
grid on; xline(best_ridge_lambda_100, 'r--');

subplot(2,2,3);
semilogx(lambda_values, lasso_coeffs_100', 'LineWidth', 0.5);
xlabel('Lambda'); ylabel('Coefficient value');
title('Lasso: 100 Basis Functions');
grid on; xline(best_lasso_lambda_100, 'r--');

% 2. Prediction plots
subplot(2,2,[2,4]);
plot(time, co2, 'k-', 'LineWidth', 2); hold on;

% Best Ridge predictions
w_ridge_100 = ridge_coeffs_100(:,best_ridge_idx_100);
plot(time_fit*max(time) + min(time), Phi_100_fit*w_ridge_100, 'b--');
plot(time_val*max(time) + min(time), Phi_100_val*w_ridge_100, 'b:');
plot(future_years, Phi_100_future*w_ridge_100, 'b-.');

% Best Lasso predictions
w_lasso_100 = [fit_info_100.Intercept(best_lasso_idx_100); lasso_coeffs_100(:,best_lasso_idx_100)];
plot(time_fit*max(time) + min(time), [ones(size(Phi_100_fit,1),1) Phi_100_fit(:,2:end)]*w_lasso_100, 'r--');
plot(time_val*max(time) + min(time), [ones(size(Phi_100_val,1),1) Phi_100_val(:,2:end)]*w_lasso_100, 'r:');

xlim([min(time) 2030]);
legend('Data', 'Ridge Train', 'Ridge Val', 'Ridge Future', 'Lasso Train', 'Lasso Val');
title('Predictions with 100 Basis Functions');
xlabel('Year'); ylabel('CO2 Concentration');

%% Helper function for 100 basis functions
function Phi = create_100_basis_functions(t)
    % t: normalized time vector (0 to 1)
    Phi = zeros(length(t), 100);
    
    % 1. Constant term
    Phi(:,1) = 1;
    
    % 2. Polynomial terms (degree 1-20)
    for i = 1:20
        Phi(:,i+1) = t.^i;
    end
    
    % 3. Trigonometric terms (21-60)
    for i = 1:40
        freq = i;
        if mod(i,2) == 1
            Phi(:,20+i) = sin(2*pi*freq*t);
        else
            Phi(:,20+i) = cos(2*pi*(freq/2)*t);
        end
    end
    
    % 4. Exponential terms (61-80)
    for i = 1:20
        Phi(:,60+i) = exp(-i*t);
    end
    
    % 5. Step functions (81-100)
    for i = 1:20
        threshold = i/21;
        Phi(:,80+i) = double(t > threshold);
    end
end