
%% 0: Clean up
clear all;
clc;
close all;


%% 1: Set file path
dir_home     = 'C:\Users\minamitu\OneDrive - University of Cincinnati\_CLASS\2-1_IndustrialAI\Final\';
toolbox      = '.\somtoolbox';

% ### dont need for this problem
% dir_train    = 'C:\Users\minamitu\OneDrive - University of Cincinnati\Class\IndustrialAI\Final\Training_DataSet';
% dir_test     = 'C:\Users\minamitu\OneDrive - University of Cincinnati\Class\IndustrialAI\Final\Testing_DataSet';
% dir_train_f1 = 'C:\Users\minamitu\OneDrive - University of Cincinnati\Class\IndustrialAI\Final\Training\Faulty\Unbalance 1';
% dir_train_f2 = 'C:\Users\minamitu\OneDrive - University of Cincinnati\Class\IndustrialAI\Final\Training\Faulty\Unbalance 2';

%% 2: Read training data

%
cd(dir_home)

% load mat file
load("train/Baseline1DataSet.mat")
load("train/Baseline2DataSet.mat")
load("train/Baseline3DataSet.mat")

load("test/TestCase1DataSet.mat")
load("test/TestCase2DataSet.mat")
load("test/TestCase3DataSet.mat")

N_train = 75; % 25+25+25 runs (from data)
N_test  = 48; % 18+15+15 runs (from data)

% joint all data
data_train = [Baseline1Run;Baseline2Run;Baseline3Run];
data_test  = [TestCase1Run;TestCase2Run;TestCase3Run];

% 
Fs = 1000;                      % Sampling frequency (from instruction)
Ts = 1/Fs;                      % Sampling period
L = 106;                        % Length of signal (from data)
t = (0:L-1)*Ts;                 % Time vector

% joint all data
data_train = [Baseline1Run;Baseline2Run;Baseline3Run];


% Visualize one data from Baseline1
train = data_train{1,1}.Data;
figure;
for ii = 2:21
    subplot(4,5,ii-1)
    plot(t, train(:,ii),'b')
    %xlabel('Time (s)')
    xlim([0.0 0.11])
    %ylabel('Amplitude')
    %ylim([-0.5 0.5])
    title('Raw Signal - Baseline1 #1')
end

%% X: Pre-processing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We should check raw sensor data
% then eliminate abnormal data and missing data etc. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%C = cell2mat(Baseline1Run);
%DT = isnan(train);


%% 3: Feature extraction of training data

N_feat = 5*19;        % Number of features

% % Lable of features
% L_feat = ...
%     ["Mean" "STD" "Minimum" "Maximum" ...
%      "Kurtosis"]; 
% x = categorical(...
%     ["Mean" "STD" "Minimum" "Maximum" ...
%      "Kurtosis"]);

 
 
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
        FeatMat_train(ii,10*(jj-1)+5) = kurtosis(data_i1);   % feature 5: Kurtosis

        FeatMat_train(ii,10*(jj-1)+6) = mean(data_i2);       % feature 1: Mean
        FeatMat_train(ii,10*(jj-1)+7) = std(data_i2);        % feature 2: STD
        FeatMat_train(ii,10*(jj-1)+8) = min(data_i2);        % feature 3: Minimum
        FeatMat_train(ii,10*(jj-1)+9) = max(data_i2);        % feature 4: Maximum 
        FeatMat_train(ii,10*(jj-1)+10)= kurtosis(data_i2);   % feature 5: Kurtosis

    end
end

FeatMat_train(isnan(FeatMat_train))=0; % Replace NaN with 0


%% 4: Dimension Reduction(PCA)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To Be done
% one of the most challenging part

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 5: Plot & compare Features

% for ii = 1:N_feat
%     figure;
%     plot(FeatMat_train(1:Nh,ii),'b-*')
%     hold on
%     plot(FeatMat_train(Nh+1:Nh+Nf1,ii),'g:s')
%     hold on
%     plot(FeatMat_train(Nh+Nf1+1:N_train,ii),'r-o')
%     hold off
%     ylabel('Amplitude')
%     xlabel('# of samples')
%     legend('healthy', 'Unbalanced 1','Unbalanced 2' )
%     title('Feature',L_feat(ii))
% end


%% 6: Delete features

% FeatMat_train(:, 3:9) = [];
% N_feat = 2;


%% 7: Normalizing(TRAINING data)

% FeatMat_mean = zeros(N_feat);                   % Mean of each features
% FeatMat_std = zeros(N_feat);                    % STD of each feature
% FeatMat_train_norm = zeros(N_train, N_feat);    % Normalized data (training)
% 
% for ii = 1:N_feat
% 
%     FeatMat_mean(ii) = mean(FeatMat_train(:,ii));   % Calc. mean
%     FeatMat_std(ii) = std(FeatMat_train(:,ii));     % Calc. std
% 
%     % Normalizing
%     FeatMat_train_norm(:,ii) = ...
%         (FeatMat_train(:,ii) - FeatMat_mean(ii))/FeatMat_std(ii);
% 
% end


%% 8: TEST DATA > Read

% cd(dir_test)
% files = dir('**/*.txt') ;   % you are in the folder of files
% N_test = length(files) ;
% 
% % loop for each file
% for ii = 1:N_test
% 
%     filename = files(ii).name;
%     temp = readtable(filename);
%     temp = table2array(temp);
% 
%     data_test(:,ii) = temp;
% 
% end


%% 9: Feature extraction(TEST data) 

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
        FeatMat_test(ii,10*(jj-1)+5) = kurtosis(data_i1);   % feature 5: Kurtosis

        FeatMat_test(ii,10*(jj-1)+6) = mean(data_i2);       % feature 1: Mean
        FeatMat_test(ii,10*(jj-1)+7) = std(data_i2);        % feature 2: STD
        FeatMat_test(ii,10*(jj-1)+8) = min(data_i2);        % feature 3: Minimum
        FeatMat_test(ii,10*(jj-1)+9) = max(data_i2);        % feature 4: Maximum 
        FeatMat_test(ii,10*(jj-1)+10)= kurtosis(data_i2);   % feature 5: Kurtosis
    end
end


%% 10: Normalizing(TEST data) 

% FeatMat_test_norm = zeros(N_test, N_feat);
% 
% for ii = 1:N_feat
% 
%     % Normalizing (using mean and std of training data)
%     FeatMat_test_norm(:,ii) = ...
%         (FeatMat_test(:,ii) - FeatMat_mean(ii))/FeatMat_std(ii);
% 
% end


%% Training Data > Labeling

% true_lables = cell(N_train,1);
% 
% for i = 1:Nh
%     true_lables{i,1} = 'H';
% end
% for i = Nh+1:Nh+Nf1
%     true_lables{i,1} = 'U1';
% end
% for i = Nh+Nf1+1:N_train
%     true_lables{i,1} = 'U2';
% end

%%% temp --------------------

FeatMat_train_norm = FeatMat_train;
FeatMat_test_norm  = FeatMat_test;

%%% temp --------------------

%% A. SOM-MQE

cd(toolbox)
% clc;


% A1. Training 
sD_1 = som_data_struct(FeatMat_train_norm);
sM_1 = som_make(sD_1);


% A2. Tesing
% true_labels = cell(N_test,1);
% 
% for i = 1:10
%     true_labels{i,1} = 'H';
% end
% for i = 11:20
%     true_labels{i,1} = 'U1';
% end
% for i = 21:30
%     true_labels{i,1} = 'U2';
% end



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

figure;
plot(MQE_train_1(:,1),"--o");
title("MQE:train");

figure;
plot(MQE_test_1(:,1),":*r");
title("MQE:test");

figure;
plot(MQE_train_n1(:,1),"--o");
title("CV:train");

figure;
plot(MQE_test_n1(:,1),":*r");
title("CV:test");


%% plot confusion matrix
% figure;

% ============================================================

% Method 1
% for k = 1:length(true_labels)
%     switch true_labels{k}
%         case 'H';   truth1(k) = 1;
%         case 'U1';  truth1(k) = 2;
%         case 'U2';  truth1(k) = 3;
%     end
% end
% truth1 = categorical( truth1 ); 

% ============================================================

% for k = 1:length(test_predict)
%     if(MQE_test_n1(k,1) > 0.925) predict1(k)=1;
%     elseif(MQE_test_n1(k,1) > 0.6) predict1(k)=2;
%     else predict1(k)=3;
%     end
% end
% predict1 = categorical( predict1 );
% 
% %
% plotconfusion(truth,predict1)
% title('SOM - Confusion Matrix')

