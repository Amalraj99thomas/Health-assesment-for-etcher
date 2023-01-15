clear; clc; close all;


%% Training Data Processing
% Load in training data for etcher 1
load('Baseline1DataSet.mat');

% Build Initial Feature Matrix
for k=1:length(Baseline1Run)
    for j=3:length(Baseline1Run{k}.Data(1, :))
        featIn(k, 9*(j-2)-8) = max(Baseline1Run{k}.Data(:, j));
        featIn(k, 9*(j-2)-7) = min(Baseline1Run{k}.Data(:, j));
        featIn(k, 9*(j-2)-6) = mean(Baseline1Run{k}.Data(:, j));
        featIn(k, 9*(j-2)-5) = kurtosis(Baseline1Run{k}.Data(:, j));
        featIn(k, 9*(j-2)-4) = std(Baseline1Run{k}.Data(:, j));
        featIn(k, 9*(j-2)-3) = var(Baseline1Run{k}.Data(:, j));
        featIn(k, 9*(j-2)-2) = rms(Baseline1Run{k}.Data(:, j));
        featIn(k, 9*(j-2)-1) = skewness(Baseline1Run{k}.Data(:, j));
        featIn(k, 9*(j-2)-0) = peak2peak(Baseline1Run{k}.Data(:, j));
    end
end
% Remove RF_Btm_Rfl_Pwr variable due to bad feature values
featIn(:, 28:36) = [];

% Perform PCA Dimensionality Reduction
[coeff, score, latent, tsquared, explained, mu] = pca(featIn);

%Biplot of first three principal components
h = biplot(coeff(:,1:3),'Scores',score(:,1:3));
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');

% Plot percentage variance explained
figure();
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');

% Reduced feature matrix
featReduced=featIn*coeff(:,1:2); % Taking only first 2 principal components

%% Testing Data Processing
% Load in training data for etcher 1
load('TestCase1DataSet.mat');

% Build Initial Feature Matrix
for k=1:length(TestCase1Run)
    for j=3:length(TestCase1Run{k}.Data(1, :))
        test_featIn(k, 9*(j-2)-8) = max(TestCase1Run{k}.Data(:, j));
        test_featIn(k, 9*(j-2)-7) = min(TestCase1Run{k}.Data(:, j));
        test_featIn(k, 9*(j-2)-6) = mean(TestCase1Run{k}.Data(:, j));
        test_featIn(k, 9*(j-2)-5) = kurtosis(TestCase1Run{k}.Data(:, j));
        test_featIn(k, 9*(j-2)-4) = std(TestCase1Run{k}.Data(:, j));
        test_featIn(k, 9*(j-2)-3) = var(TestCase1Run{k}.Data(:, j));
        test_featIn(k, 9*(j-2)-2) = rms(TestCase1Run{k}.Data(:, j));
        test_featIn(k, 9*(j-2)-1) = skewness(TestCase1Run{k}.Data(:, j));
        test_featIn(k, 9*(j-2)-0) = peak2peak(TestCase1Run{k}.Data(:, j));
    end
end
% Remove RF_Btm_Rfl_Pwr variable due to bad feature values
test_featIn(:, 28:36) = [];

% Perform PCA Dimensionality Reduction
[test_coeff, test_score, test_latent, test_tsquared, test_explained, test_mu] = pca(test_featIn);

% Reduced feature matrix
test_featReduced=test_featIn*test_coeff(:,1:2); % Taking only first 2 principal components

%% SOM MQE Health Determination
% Add SOM toolbox path
addpath('F:\DOCUMENTS\School\2021 Spring\Industrial Big Data and AI\HW3\somtoolbox');

% Create SOM
sM=som_make(featReduced);
S=size(test_featReduced);
S=S(1);

for i=1:S
    qe=som_quality(sM,test_featReduced(i,:)); % calculate MQE value for each sample
    MQEt(i)=qe;
end
MQEtn=((MQEt)./(max(MQEt))); % normalize MQE
MQEtn=MQEtn';

%Plot MQE
figure();
plot(MQEtn,'-');
xlabel('Data file No.');
ylabel('MQE Value');
title('MQE Health Assessment Plot');
