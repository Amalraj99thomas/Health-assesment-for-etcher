
%% This code was tested on MATLAB R2021b


%% 00: Clean up
clear all;
clc;
close all;


%% 01: Set file path
dir_home     = '\\clusterfsnew.ceas1.uc.edu\students\thoma6aj\desktop\Project_main';
toolbox      = '.\somtoolbox';


%% 02: Read training data and testing data

% set dir
cd(dir_home)

% load mat file(train and test)
load("train/Baseline1DataSet.mat")
load("train/Baseline2DataSet.mat")
load("train/Baseline3DataSet.mat")

load("test/TestCase1DataSet.mat")
load("test/TestCase2DataSet.mat")
load("test/TestCase3DataSet.mat")

% A. Use all data

% joint all data
data_train = [Baseline1Run;Baseline2Run;Baseline3Run];
data_test  = [TestCase1Run;TestCase2Run;TestCase3Run];

% set parameters
N_train = 75; % 25+25+25 runs (from data)
N_test  = 48; % 18+15+15 runs (from data)


% B. Use only Baseline1

%data_train = Baseline1Run;
%data_test  = TestCase1Run;
%N_train = 25;
%N_test  = 18;


%% Visualize raw data

% Fs = 1000;                      % Sampling frequency (from instruction)
% Ts = 1/Fs;                      % Sampling period
% L = 106;                        % Length of signal (from data)
% t = (0:L-1)*Ts;                 % Time vector

% Visualize one data from Baseline1
train1 = data_train{1,1}.Data;
figure;
for ii = 2:21
    subplot(4,5,ii-1)
    plot(train1(:,ii),'b')
    %xlabel('Time (s)')
    xlim([0.0 0.11])
    %ylabel('Amplitude')
    %ylim([-0.5 0.5])
    title(ii)
end

%% Debug: Visualize and Compare raw data
train2 = data_train{19,1}.Data;
figure;
for ii = 2:21
    subplot(4,5,ii-1)
    plot(train1(:,ii),'b')
    hold on;
    plot(train2(:,ii),'r')
    title("Raw data")
    %title(ii)
end

%% Visualize raw data (TEST)
train1 = data_test{29,1}.Data;
train2 = data_test{28,1}.Data;
figure;
for ii = 2:21
    %% subplot(4,5,ii-1)
    %% plot(train1(:,ii),'b')
    hold on;
    %plot(train2(:,ii),'r')
    %title(ii)
end

%% Visualize raw data (TEST)

figure;
for ii = 1:15
    train1 = data_test{18+ii,1}.Data;
    %% plot(train1(:,6))
    hold on;
end
title("sensor6")


%% XX: Pre-processing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To Be Done

% We should check raw sensor data
% then eliminate abnormal data and missing data etc. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 03: Feature extraction (training data)

N_feat = 10*19;        % Number of features

FeatMat_train = zeros(N_train, N_feat); % Feature Matrix of training data

N = N_train;
for ii = 1:N
    data_all = data_train{ii,1}.Data; % SETTING
    
    for jj = 1:19 % Use 19 sensors data

        row = find(data_all(:,2)==4);
        data_i1 = data_all(row,jj+2); % Use #3 to #21 from data
        
        row = find(data_all(:,2)==5);
        data_i2 = data_all(row,jj+2); % Use #3 to #21 from data

        FeatMat_train(ii,10*(jj-1)+1) = mean(data_i1);       % feature 1: Mean
        FeatMat_train(ii,10*(jj-1)+2) = std(data_i1);        % feature 2: STD
        FeatMat_train(ii,10*(jj-1)+3) = min(data_i1);        % feature 3: Minimum
        FeatMat_train(ii,10*(jj-1)+4) = max(data_i1);        % feature 4: Maximum 
        %FeatMat_train(ii,10*(jj-1)+5) = peak2peak(data_i1);   % feature 5: Peak-to-Peak
        FeatMat_train(ii,10*(jj-1)+5) = kurtosis(data_i1);   % feature 5: Kurtosis

        FeatMat_train(ii,10*(jj-1)+6) = mean(data_i2);       % feature 1: Mean
        FeatMat_train(ii,10*(jj-1)+7) = std(data_i2);        % feature 2: STD
        FeatMat_train(ii,10*(jj-1)+8) = min(data_i2);        % feature 3: Minimum
        FeatMat_train(ii,10*(jj-1)+9) = max(data_i2);        % feature 4: Maximum 
        %FeatMat_train(ii,10*(jj-1)+10)= peak2peak(data_i1);   % feature 5: Peak-to-Peak
        FeatMat_train(ii,10*(jj-1)+10)= kurtosis(data_i2);   % feature 5: Kurtosis

    end
end

%% 04: Feature extraction(TEST data) 
N_feat = 10*19;        % Number of features

FeatMat_test = zeros(N_test, N_feat);
N = N_test;

for ii = 1:N
    data_all = data_test{ii,1}.Data; % SETTING
    
    for jj = 1:19 % Use #3 to #21 from data

        row = find(data_all(:,2)==4);
        data_i1 = data_all(row,jj+2); % Use #3 to #21 from data
        
        row = find(data_all(:,2)==5);
        data_i2 = data_all(row,jj+2); % Use #3 to #21 from data

        FeatMat_test(ii,10*(jj-1)+1) = mean(data_i1);       % feature 1: Mean
        FeatMat_test(ii,10*(jj-1)+2) = std(data_i1);        % feature 2: STD
        FeatMat_test(ii,10*(jj-1)+3) = min(data_i1);        % feature 3: Minimum
        FeatMat_test(ii,10*(jj-1)+4) = max(data_i1);        % feature 4: Maximum 
        % FeatMat_test(ii,10*(jj-1)+5) = peak2peak(data_i1);   % feature 5: Peak-to-peak
        FeatMat_test(ii,10*(jj-1)+5) = kurtosis(data_i1);   % feature 5: Kurtosis

        FeatMat_test(ii,10*(jj-1)+6) = mean(data_i2);       % feature 1: Mean
        FeatMat_test(ii,10*(jj-1)+7) = std(data_i2);        % feature 2: STD
        FeatMat_test(ii,10*(jj-1)+8) = min(data_i2);        % feature 3: Minimum
        FeatMat_test(ii,10*(jj-1)+9) = max(data_i2);        % feature 4: Maximum 
        % FeatMat_test(ii,10*(jj-1)+10)= peak2peak(data_i2);   % feature 5: Peak-to-peak
        FeatMat_test(ii,10*(jj-1)+10)= kurtosis(data_i2);   % feature 5: Kurtosis

    end
end

%% TEMP: pre-processing

FeatMat_train(isnan(FeatMat_train))=0; % Replace NaN with 0

%% Visualize Features

for jj = 1:19
    s = zeros(19);
    m= zeros(19);
    figure;
    for ii = 1:10
        kk = ii + 10*(jj-1);
        %subplot(2,5,ii)
        %xlabel('Time (s)')
        %xlim([0.0 0.11])
        %ylabel('Amplitude')
        %ylim([-0.5 0.5])
        %plot(FeatMat_train(:,kk),'b')
        %hold on
        %plot(FeatMat_test(:,kk),'r')
        %title(kk)
        %figure;
        
    end
end

disp(length(FeatMat_train))


for jj = 1:190
    %not a normal distribution.
    s = zeros(190);
    m= zeros(190);
    figure;
    for ii = 1:10
        kk = ii + 10*(jj-1);
        disp(length(FeatMat_train(jj)))
        histogram(FeatMat_train(:,kk))
        title("Check")
        S = std(FeatMat_train(:,kk));
        s(jj) = S;
        disp(s)
        disp(m)
    end
end



%% 05: Normalizing(TRAINING data)
 
FeatMat_mean = zeros(N_feat,1);                 % Mean of each features
FeatMat_std  = zeros(N_feat,1);                 % STD of each feature
FeatMat_train_norm = zeros(N_train, N_feat);    % Normalized data (training)

for ii = 1:N_feat

    FeatMat_mean(ii,1) = mean(FeatMat_train(:,ii));   % Calc. mean
    FeatMat_std(ii,1)  =  std(FeatMat_train(:,ii));   % Calc. std

    if FeatMat_std(ii,1) == 0
        FeatMat_std(ii,1) = 1;
        display(ii)
    end

    % Normalizing
    FeatMat_train_norm(:,ii) = ...
        (FeatMat_train(:,ii) - FeatMat_mean(ii,1))/FeatMat_std(ii,1);

end

% Z5 = sum(FeatMat_train_norm); % debug


%% 06: Normalizing(TEST data) 

FeatMat_test_norm = zeros(N_test, N_feat);

for ii = 1:N_feat

    % Normalizing (using mean and std of training data)
    FeatMat_test_norm(:,ii) = ...
        (FeatMat_test(:,ii) - FeatMat_mean(ii,1))/FeatMat_std(ii,1);

end

Y5 = sum(FeatMat_train_norm); % debug


%% Visualize > One Normalized data from Baseline1(train)

% for jj = 1:19
%     figure;
%     for ii = 1:10
%         kk = ii + 10*(jj-1);
%         subplot(2,5,ii)
%         plot(FeatMat_train_norm(:,kk),'b')
%         hold on
%         plot(FeatMat_test_norm(:,kk),'r')
%         title(kk)
%     end
% end


%% TEMP: Pre-processing

% 1.Eliminate some features
%del_idx = [33, 38, 54, 163, 168, 169, 188];
%del_idx = [33, 38, 168];
%FeatMat_train_norm(:, del_idx) = [];
%FeatMat_test_norm(:, del_idx) = [];

% 2.Replace NaN to "0"
FeatMat_train_norm(isnan(FeatMat_train_norm))=0; % Replace NaN with 0
FeatMat_test_norm(isnan(FeatMat_test_norm))=0; % Replace NaN with 0


%% 07: Dimension Reduction(PCA)

% ---------------------------
% code from IAI.txt
% ---------------------------

AA = FeatMat_train_norm
BB = FeatMat_test_norm 

% Find the principal components for the training data 
[coeff1,score1,latent,tsquared,explained,mu] = pca(AA); 

% Display the percent variability
explained

figure;
plot(explained)

figure;
plot(cumsum(explained))

% Find the number of components required to explain at least 95% variability.
idx = find(cumsum(explained)>95,1); 
scoreTrain95 = score1(:,1:idx);

% =========
NN = 10; % # of selected principal components
% =========

[coeff2,scoreTrain,latent,tsquared,explained,mu] = pca(AA,'NumComponents',NN);

% debug1 = FeatMat_train_norm*coeff2; % same scoreTrain

% coeff2 = X mxr
scoreTest = BB*coeff2;

%% ####################################################
% A. PCA-SPE
% -----------------------------------------------------

for ii = 1:N_train
    PCr(ii,:) = AA(ii,:) * coeff2;
    Em(ii,:)  = AA(ii,:) - PCr(ii,:) * transpose(coeff2);
    SPE_train(ii) = dot(Em(ii,:),Em(ii,:));
end

for ii = 1:N_test
    PCr(ii,:) = BB(ii,:) * coeff2;
    Em(ii,:)  = BB(ii,:) - PCr(ii,:) * transpose(coeff2);
    SPE_test(ii) = dot(Em(ii,:),Em(ii,:));
end

figure;
SPE = [SPE_train.'; SPE_test.'];
plot(SPE);
title("SPE");


%% ####################################################
% B. PCA-T2 
% -----------------------------------------------------

tt_train = mahal(scoreTrain,scoreTrain);
tt_test  = mahal(scoreTest,scoreTest);

tt = [tt_train; tt_test];
figure;
plot(tt);
title("T2");


%% ####################################################
% % PLOT
% scatter3(scoreTrain95(:,1),scoreTrain95(:,2),scoreTrain95(:,3))
% axis equal
% xlabel('1st Principal Component')
% ylabel('2nd Principal Component')
% zlabel('3rd Principal Component')





%% (Eliminate the features which only have "0" value.)

% FeatMat_train(isnan(FeatMat_train))=0; % Replace NaN with 0
% 
% % trick
% FeatMat_train(1,:) = FeatMat_train(1,:) + 0.001;
% 

% del_train1 = find(max(FeatMat_train)==min(FeatMat_train));
% del_train2 = find(isnan(sum(FeatMat_train)))

% 
% % FeatMat_train(:, del_train) = [];
% % N_feat = N_feat - numel(del_train); 
% % 
% % Z1 = sum(FeatMat_train);
% % Z2 = max(FeatMat_train);
% % Z3 = min(FeatMat_train);
% % Z4 = cat(1,Z1,Z2,Z3);


%% XX: Eliminate the features which only have "0" value.

% FeatMat_test(isnan(FeatMat_test))=0; % Replace NaN with 0
% 
% % trick
% FeatMat_test(1,:) = FeatMat_test(1,:) + 0.001;
% 
% % Y1 = sum(FeatMat_test);
% % Y2 = max(FeatMat_test);
% % Y3 = min(FeatMat_test);
% % Y4 = cat(1,Y1,Y2,Y3);
% 
% del_test = find(max(FeatMat_test)==min(FeatMat_test));
% del_test2= find(isnan(sum(FeatMat_test)))

% 
% % FeatMat_test(:, del_test) = [];
% % N_feat = N_feat - numel(del_test);        % TEMP




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % PCA Option-1
% 
% % modify the feature matrix
% FeatMat_test_norm_M = FeatMat_test_norm * vec;
% 
% % Eliminate unnecessary modified features (Cumilative Variance > 95%)
% TEST = FeatMat_test_norm_M(:,:);
% FeatMat_test_norm_M(:, 1:end-8) = [];

%% XX: temp.

FeatMat_train_norm = scoreTrain;
FeatMat_test_norm  = scoreTest;


%% A0. SOM-MQE

cd(toolbox)
% clc;


% A1. Training 
sD_1 = som_data_struct(FeatMat_train_norm);
sM_1 = som_make(sD_1);


% A2. MQE
for k = 1:size(FeatMat_train_norm, 1)
    qe = som_quality(sM_1,FeatMat_train_norm(k,:));
    MQE_train_1(k,1) = qe;
end

for k = 1:size(FeatMat_test_norm, 1)
    qe = som_quality(sM_1,FeatMat_test_norm(k,:));
    MQE_test_1(k,1) = qe;
end

MQE_max_1 = max(MQE_train_1);
MQE_train_n1 = 1-MQE_train_1/MQE_max_1;
MQE_test_n1  = 1-MQE_test_1/MQE_max_1;

% A3. plot
figure;
plot(MQE_train_1(:,1),"--o");
title("MQE:train");
figure;
plot(FeatMat_train_norm);
title("FeatMat_train_norm");

figure;
%% plot(MQE_test_1(:,1),":*r");
title("MQE:test");

figure;
%% plot(MQE_train_n1(:,1),"--o");
title("CV:train");

figure;
%% plot(MQE_test_n1(:,1),":*r");
title("CV:test");

%% Output prediction

% To Be Done

%% Test something



