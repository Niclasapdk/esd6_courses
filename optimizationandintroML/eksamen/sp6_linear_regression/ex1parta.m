%% Load and prepare data (assuming data is loaded as time and co2 vectors)
data = import_co2_concentration('co2_weekly_mlo.txt');
time = data.year;
co2 = data.co2_ppm;
% Normalize time to [0,1] for better numerical stability
time_normalized = (time - min(time)) / (max(time) - min(time));

% Split data into first half for fitting, second half for validation
split_idx = floor(length(time)/2);
time_fit = time_normalized(1:split_idx);
co2_fit = co2(1:split_idx);
time_val = time_normalized(split_idx+1:end);
co2_val = co2(split_idx+1:end);

%% Create basic basis functions matrix (5 functions)
Phi_fit = [ones(length(time_fit), 1), ...              % φ1 = 1
           time_fit, ...                               % φ2 = t
           time_fit.^2, ...                            % φ3 = t²
           sin(2*pi*time_fit), ...                     % φ4 = sin(2πt)
           cos(2*pi*time_fit)];                        % φ5 = cos(2πt)

%% Ridge Regression (closed-form solution)
lambda_ridge = 0.1; % Regularization parameter
I = eye(size(Phi_fit, 2)); % Identity matrix
w_ridge = (Phi_fit' * Phi_fit + lambda_ridge * I) \ (Phi_fit' * co2_fit);

% Evaluate on validation set
Phi_val = [ones(length(time_val), 1), time_val, time_val.^2, ...
           sin(2*pi*time_val), cos(2*pi*time_val)];
co2_pred_ridge = Phi_val * w_ridge;

%% Lasso Regression using lasso function (Statistics and Machine Learning Toolbox)
[w_lasso, fit_info] = lasso(Phi_fit(:,2:end), co2_fit, 'Lambda', 0.1, 'Intercept', true);
% Note: lasso function automatically includes intercept, so we remove the
% first column of ones(base funciton 1)

%% Define lambda range (critical for the exercise)
lambda_values = logspace(-4, 2, 50); % 50 values from 10^-4 to 100

%% Ridge Regression: Test all lambda values
ridge_val_errors = zeros(size(lambda_values));
ridge_coeffs = zeros(size(Phi_fit,2), length(lambda_values));

for i = 1:length(lambda_values)
    % Current lambda
    lambda = lambda_values(i);
    
    % Solve ridge regression
    w = (Phi_fit'*Phi_fit + lambda*eye(size(Phi_fit,2))) \ (Phi_fit'*co2_fit);
    
    % Store coefficients
    ridge_coeffs(:,i) = w;
    
    % Calculate validation error
    pred = Phi_val * w;
    ridge_val_errors(i) = sqrt(mean((pred - co2_val).^2));
end

% Find optimal lambda
[best_ridge_error, best_idx] = min(ridge_val_errors);
best_ridge_lambda = lambda_values(best_idx);

%% Lasso Regression: Test lambda values
% If using MATLAB's lasso() function:
[lasso_coeffs, fit_info] = lasso(Phi_fit(:,2:end), co2_fit, 'Lambda', lambda_values);

% Calculate validation errors
lasso_val_errors = zeros(size(lambda_values));
for i = 1:length(lambda_values)
    pred = fit_info.Intercept(i) + Phi_val(:,2:end)*lasso_coeffs(:,i);
    lasso_val_errors(i) = sqrt(mean((pred - co2_val).^2));
end

% Find optimal lambda
[best_lasso_error, best_lasso_idx] = min(lasso_val_errors);
best_lasso_lambda = lambda_values(best_lasso_idx);

%% Display results
fprintf('Optimal lambda values:\n');
fprintf('Ridge: %.4f (validation RMSE: %.2f)\n', best_ridge_lambda, best_ridge_error);
fprintf('Lasso: %.4f (validation RMSE: %.2f)\n', best_lasso_lambda, best_lasso_error);

%% Define labels for basis functions
basis_labels = {'1', 't', 't^2', 'sin(2\pit)', 'cos(2\pit)'};

%% Plot lambda analysis
figure;

% Ridge coefficients path
subplot(2,2,1);
semilogx(lambda_values, ridge_coeffs', 'LineWidth', 1.5);
xlabel('Lambda');
ylabel('Coefficient value');
title('Ridge Coefficient Paths');
grid on;
hold on;
xline(best_ridge_lambda, 'r--', 'LineWidth', 1.5);
legend(basis_labels, 'Location', 'best');

% Ridge validation error
subplot(2,2,2);
loglog(lambda_values, ridge_val_errors, 'b-o', 'LineWidth', 1.5);
xlabel('Lambda');
ylabel('Validation RMSE');
title('Ridge Validation Error');
grid on;
hold on;
plot(best_ridge_lambda, best_ridge_error, 'ro', 'MarkerSize', 10);

% Lasso coefficients path
subplot(2,2,3);
semilogx(lambda_values, lasso_coeffs', 'LineWidth', 1.5);
xlabel('Lambda');
ylabel('Coefficient value');
title('Lasso Coefficient Paths');
grid on;
hold on;
xline(best_lasso_lambda, 'r--', 'LineWidth', 1.5);
legend(basis_labels(2:end), 'Location', 'best');  % Lasso excludes φ1 (intercept)

% Lasso validation error
subplot(2,2,4);
loglog(lambda_values, lasso_val_errors, 'r-o', 'LineWidth', 1.5);
xlabel('Lambda');
ylabel('Validation RMSE');
title('Lasso Validation Error');
grid on;
hold on;
plot(best_lasso_lambda, best_lasso_error, 'bo', 'MarkerSize', 10);
