%% PCA via SVD on Fisher’s Iris (no built‑in pca)
load fisheriris             % meas (150×4), species (150×1 cell)
X = meas';                  % d×N = 4×150
labels = species;           % cell array of 3 classes

% Center
N  = size(X,2);

% Center the data to obtain Xc 

X_bar = mean(X, 2);         % Mean of each row (feature) %FILL MISSING CODE
Xc = X - X_bar;             % Centered data%FILL MISSING CODE


% SVD to perform PCA: 
[U, S, V] = svd(Xc, 'econ');   % Xc = U * S * V'%FILL THE MISSING CODE (use svd function with matlab)

%The eigenvalues of the covariance matrix as a column vector
lambda = diag(S).^2 / (N - 1);  % Eigenvalues (variance explained)%FILL MISSING CODE
%Individual explained variance:
explained = lambda / sum(lambda) * 100;%FILL MISSING CODE
%Cummulative explained variance: 
cumExplained  = cumsum(explained);

%PC componenets: 
U = U;% Principal directions (already obtained)%FILL MISSING CODE

% Coordinates (scores or codes)
Z = U' * Xc;% Project data onto PCs%FILL MISSING CODE;

%Reconstruction (projection) of X:
Xc_hat = U * Z;                % Reconstruct centered data%FILL MISSING CODE 
X_hat = Xc_hat + X_bar;        % Reconstruct original data%FILL MISSING CODE

% Unique labels
labs = unique(labels);  % {'setosa','versicolor','virginica'}

%% Explained variance plot
figure;
bar(explained,'FaceAlpha',0.6); hold on
plot(cumExplained,'r-o','LineWidth',1.5);
hold off
xlabel('Principal Component'); ylabel('Variance Explained (%)');
title('Iris: Explained Variance by PC (via SVD)');
legend('Individual','Cumulative','Location','Best'); grid on;

%% 2D scatter PC1 vs PC2
figure; hold on;
markers = {'o','s','^'};
colors  = lines(3);
for c = 1:numel(labs)
    idx = strcmp(labels, labs{c});
    scatter(Z(1,idx), Z(2,idx), 36, colors(c,:), markers{c}, 'filled');
end
hold off
xlabel('PC1'); ylabel('PC2');
title('Iris in PC1–PC2 Space (via SVD)');
legend(labs,'Location','Best'); grid on;

%% 3D scatter PC1–PC3
figure; hold on;
for c = 1:numel(labs)
    idx = strcmp(labels, labs{c});
    scatter3(Z(1,idx), Z(2,idx), Z(3,idx), 36, colors(c,:), markers{c}, 'filled');
end
hold off
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('Iris in PC1–PC2–PC3 Space (via SVD)');
legend(labs,'Location','Best'); grid on; view(-30,10);

%% Feature loadings
featureNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth'};
figure;
bar(U(:,1:2));
set(gca,'XTick',1:4,'XTickLabel',featureNames,'XTickLabelRotation',45);
ylabel('Loading'); title('Feature Loadings for PC1 & PC2 (via SVD)');
legend('PC1','PC2','Location','Best'); grid on;

%% Reconstruction error vs #PCs
maxK = size(U,2);
mse  = zeros(maxK,1);
for k = 1:maxK
    Xc_hat = U(:,1:k)*(U(:,1:k)'*Xc);
    mse(k) = mean(sum((Xc - Xc_hat).^2,1));
end
figure;
plot(1:maxK, mse,'-o','LineWidth',1.5);
xlabel('Number of PCs'); ylabel('Reconstruction MSE');
title('Iris: Reconstruction Error vs. # of PCs (via SVD)');
grid on;
