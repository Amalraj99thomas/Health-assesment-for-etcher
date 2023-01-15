
%% 00: Clean up
clear all;
clc;
close all;


%% This code was tested on MATLAB R2021b

% a=[0
% 1
% 0
% 1
% 0
% 0
% 1
% 0
% 1
% 0
% 1
% 1
% 0
% 1
% 0
% 1
% 1
% 0
% 1
% 0
% 0
% 1
% 0
% 1
% 1
% 0
% 0
% 0
% 1
% 0
% 0
% 0
% 1
% 0
% 1
% 1
% 0
% 0
% 1
% 0
% 0
% 1
% 0
% 0
% 1
% 1
% 0
% 0
% ];

a1=[0
1
0
1
0
0
1
0
1
0
1
1
0
1
0
1
1
0
];

a2=[1
0
0
1
0
1
1
0
0
0
1
0
0
0
1
];

a3=[
0
1
1
0
0
1
0
0
1
0
0
1
1
0
0
];

a = a2;



%% 01: Set 
toolbox      = '.\somtoolbox';
flag = 1; % visualize 1:on, 0:off

%% 02: Read training data and testing data

% load mat file(train and test)
load("train/Baseline1DataSet.mat")
load("train/Baseline2DataSet.mat")
load("train/Baseline3DataSet.mat")

load("test/TestCase1DataSet.mat")
load("test/TestCase2DataSet.mat")
load("test/TestCase3DataSet.mat")

% A. Use all data

% % joint all data
% data_train = [Baseline1Run;Baseline2Run;Baseline3Run];
% data_test  = [TestCase1Run;TestCase2Run;TestCase3Run];
% 
% % set parameters
% N_train = 75; % 25+25+25 runs (from data)
% N_test  = 48; % 18+15+15 runs (from data)


% B. Use only Baseline1

data_train = Baseline2Run;
data_test  = TestCase2Run;
N_train = 25;
N_test  = 15;


%% Visualize raw data

% Visualize one data from Baseline1
if flag == 1
    train1 = data_train{1,1}.Data;
    figure; % Fig.1
    for ii = 2:21
        subplot(4,5,ii-1)
        plot(train1(:,ii),'b')
        title(ii)
        sgtitle("Fig.1 Raw signal (#1 of Baseline1)")
    end
end

%% Debug: Visualize and Compare raw data
% if flag == 1
%     train2 = data_train{19,1}.Data;
%     figure;
%     for ii = 2:21
%         subplot(4,5,ii-1)
%         plot(train1(:,ii),'b')
%         hold on;
%         plot(train2(:,ii),'r')
%         title(ii)
%         sgtitle("DEBUG. Raw signal ()")
% 
%     end
% end

%% Visualize raw data (TEST)
% if flag == 1
%     train1 = data_test{29,1}.Data;
%     train2 = data_test{28,1}.Data;
%     figure;
%     for ii = 2:21
%         subplot(4,5,ii-1)
%         plot(train1(:,ii),'b')
%         hold on;
%         plot(train2(:,ii),'r')
%         title(ii)
%         sgtitle("DEBUG. Raw signal ()")
% 
%     end
% end

%% Visualize raw data (TEST)
% if flag == 1
%     figure;
%     for ii = 1:15
%         train1 = data_test{18+ii,1}.Data;
%         plot(train1(:,6))
%         hold on;
%     end
%     sgtitle("DEBUG. Raw signal ()")
% end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature Extraction 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 03: Feature extraction (training data)

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
        FeatMat_train(ii,10*(jj-1)+5) = kurtosis(data_i1);   % feature 5: Kurtosis

        FeatMat_train(ii,10*(jj-1)+6) = mean(data_i2);       % feature 1: Mean
        FeatMat_train(ii,10*(jj-1)+7) = std(data_i2);        % feature 2: STD
        FeatMat_train(ii,10*(jj-1)+8) = min(data_i2);        % feature 3: Minimum
        FeatMat_train(ii,10*(jj-1)+9) = max(data_i2);        % feature 4: Maximum 
        FeatMat_train(ii,10*(jj-1)+10)= kurtosis(data_i2);   % feature 5: Kurtosis

    end
end

% ---------------------------------------------------------------------------------------------
% 04: Feature extraction(TEST data) 
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

        FeatMat_test(ii,10*(jj-1)+1) =     mean(data_i1);   % feature 1: Mean
        FeatMat_test(ii,10*(jj-1)+2) =      std(data_i1);   % feature 2: STD
        FeatMat_test(ii,10*(jj-1)+3) =      min(data_i1);   % feature 3: Minimum
        FeatMat_test(ii,10*(jj-1)+4) =      max(data_i1);   % feature 4: Maximum 
        FeatMat_test(ii,10*(jj-1)+5) = kurtosis(data_i1);   % feature 5: Kurtosis

        FeatMat_test(ii,10*(jj-1)+6) =     mean(data_i2);   % feature 1: Mean
        FeatMat_test(ii,10*(jj-1)+7) =      std(data_i2);   % feature 2: STD
        FeatMat_test(ii,10*(jj-1)+8) =      min(data_i2);   % feature 3: Minimum
        FeatMat_test(ii,10*(jj-1)+9) =      max(data_i2);   % feature 4: Maximum 
        FeatMat_test(ii,10*(jj-1)+10)= kurtosis(data_i2);   % feature 5: Kurtosis

    end
end

%% TEMP: pre-processing

% FLAG
%FeatMat_train(isnan(FeatMat_train))=0; % Replace NaN with 0

%% Visualize Features
if flag == 1

    for jj = 1:19
        figure;
        for ii = 1:10
            kk = ii + 10*(jj-1);
            subplot(2,5,ii)
            plot(FeatMat_train(:,kk),'b')
            hold on
            plot(FeatMat_test(:,kk),'r')
            title(kk)
            sgtitle("Fig. Features from sensor #"+jj)
        end
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
    end

    % Normalizing
    FeatMat_train_norm(:,ii) = ...
        (FeatMat_train(:,ii) - FeatMat_mean(ii,1))/FeatMat_std(ii,1);

end


%% 06: Normalizing(TEST data) 

FeatMat_test_norm = zeros(N_test, N_feat);

for ii = 1:N_feat

    % Normalizing (using mean and std of training data)
    FeatMat_test_norm(:,ii) = ...
        (FeatMat_test(:,ii) - FeatMat_mean(ii,1))/FeatMat_std(ii,1);

end

%% TEMP: Pre-processing

% 1.Eliminate some features
%del_idx = [33, 38, 54, 163, 168, 169, 188];
%del_idx = [33, 38, 168];
%FeatMat_train_norm(:, del_idx) = [];
%FeatMat_test_norm(:, del_idx) = [];

% 2.Replace NaN to "0"
FeatMat_train_norm(isnan(FeatMat_train_norm))=0; % Replace NaN with 0
FeatMat_test_norm(isnan(FeatMat_test_norm))=0; % Replace NaN with 0


%% ####################################################
% 07: Dimension Reduction(PCA)
% -----------------------------------------------------

AA = FeatMat_train_norm;
BB = FeatMat_test_norm; 

% Find the principal components for the training data 
[coeff1,score1,latent,tsquared,explained,mu] = pca(AA); 

% Display the percent variability
explained;

figure;
plot(explained)
title('Fig. Eigenvalues')


figure;
plot(cumsum(explained))
title('DEBUG')



% Find the number of components required to explain at least 95% variability.
idx = find(cumsum(explained)>95,1); 
scoreTrain95 = score1(:,1:idx);

% =========
NN = 10; % # of selected principal components
% =========

[coeff2,scoreTrain,latent2,tsquared,explained,mu] = pca(AA,'NumComponents',NN);

% debug1 = FeatMat_train_norm*coeff2; % same scoreTrain

% coreTrain: principle componentsp
% coeff2: eigenvectors

% coeff2 = X mxr
scoreTest = BB*coeff2;

%% ####################################################
%  ####################################################
%
% A. SPE
%
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


% Calculate threshold

SPE_mean = mean(SPE_train);   % Calc. mean
SPE_std  =  std(SPE_train);   % Calc. std
SPE_threshold = SPE_mean + 14 * SPE_std;

for ii = 1:N_test 
    if SPE_test(ii) > SPE_threshold
        SPE_predict(ii) = 1;
    else
        SPE_predict(ii) = 0;
    end
end



% FIGURE >> SPE

% ground truth


b = zeros(N_test,1)
b(:) = 1

c = b-a;

plot_threshold(:) = zeros(N_test+N_train,1);
plot_threshold(:) = SPE_threshold;
XX0 = (1:N_test+N_train);
XX1 = (1:N_train);
XX2 = (N_train+1:N_train+N_test);
YY_train   = SPE_train;
YY_faulty = SPE_test.*a'; 
YY_healthy  = SPE_test.*c';
SPE = [SPE_train.'; SPE_test.'];

figure;
semilogy(XX1,YY_train,"ob",XX2,YY_healthy,"og",XX2,YY_faulty,"or",XX0,plot_threshold,'r');
xlim([1,N_train+N_test]);
sgtitle("Fig. Result: SPE");



%% ####################################################

% B. T2 (another way to calculate)

% -----------------------------------------------------

Di = diag(latent2);
Di = Di(1:NN,1:NN);
inv_Di = inv(Di);

for ii = 1:N_train
    PCr(ii,:) = AA(ii,:) * coeff2;
    tt_train_a(ii)  = PCr(ii,:) * inv_Di * PCr(ii,:).';
end

for ii = 1:N_test
    PCr(ii,:) = BB(ii,:) * coeff2;
    tt_test_a(ii)  = PCr(ii,:) * inv_Di * PCr(ii,:).';
end

% Calculate threshold

tt_mean_a = mean(tt_train_a);   % Calc. mean
tt_std_a  =  std(tt_train_a);   % Calc. std
tt_threshold_a = tt_mean_a + 3 * tt_std_a;
tt_threshold_a = 25.51;

for ii = 1:N_test 
    if tt_test_a(ii) > tt_threshold_a
        tt_predict_a(ii) = 1;
    else
        tt_predict_a(ii) = 0;
    end
end



% figure;
% plot(tt_test_a);
% title("T2_test");

% FIGURE >> SPE
plot_threshold(:) = zeros(N_test+N_train,1);
plot_threshold(:) = tt_threshold_a;
XX0 = (1:N_test+N_train);
XX1 = (1:N_train);
XX2 = (N_train+1:N_train+N_test);
YY_train   = tt_train_a;
YY_faulty = tt_test_a.*a'; 
YY_healthy  = tt_test_a.*c';
TT_a = [tt_train_a.'; tt_test_a.'];

figure;
semilogy(XX1,YY_train,"ob",XX2,YY_healthy,"og",XX2,YY_faulty,"or",XX0,plot_threshold,'r');
xlim([1,N_train+N_test]);
sgtitle("Fig. Result: T2");




%% ####################################################

% B. T2 

% -----------------------------------------------------

tt_train = mahal(scoreTrain,scoreTrain);
tt_test  = mahal(scoreTest,scoreTest);



% Calculate threshold

tt_mean = mean(tt_train);   % Calc. mean
tt_std  =  std(tt_train);   % Calc. std
tt_threshold = tt_mean + 3 * tt_std;

for ii = 1:N_test 
    if tt_test(ii) > tt_threshold
        tt_predict(ii) = 1;
    else
        tt_predict(ii) = 0;
    end
end



% ------------------------------------------------
% Plot Results

% % Old plot
% tt = [tt_train; tt_test];
% figure;
% plot(tt);
% title("T2");

% ---
plot_threshold(:) = zeros(N_test+N_train,1);
plot_threshold(:) = tt_threshold;
XX0 = (1:N_test+N_train);
XX1 = (1:N_train);
XX2 = (N_train+1:N_train+N_test);
YY_train   = tt_train;
YY_faulty = tt_test'.*a'; 
YY_healthy  = tt_test'.*c';
tt = [tt_train; tt_test];

figure;
semilogy(XX1,YY_train,"ob",XX2,YY_healthy,"og",XX2,YY_faulty,"or",XX0,plot_threshold,'r');
xlim([1,N_train+N_test]);
sgtitle("Fig. Result: T2 (old)");


%% XX: temp.

% FeatMat_train_norm = scoreTrain;
% FeatMat_test_norm  = scoreTest;


%% A0. SOM-MQE

cd(toolbox)
% clc;


% A1. Training 
sD_1 = som_data_struct(scoreTrain);
sM_1 = som_make(sD_1);


% A2. MQE
for k = 1:size(scoreTrain, 1)
    qe = som_quality(sM_1,scoreTrain(k,:));
    MQE_train_1(k,1) = qe;
end

for k = 1:size(scoreTest, 1)
    qe = som_quality(sM_1,scoreTest(k,:));
    MQE_test_1(k,1) = qe;
end

% -------------------------------------------------------------

% figure;
% plot(MQE_train_1(:,1),"--o");
% title("Debug. MQE:train");

% -------------------------------------------------------------

% figure;
% plot(MQE_test_1(:,1),":*r");
% title("Debug. MQE:test");

% -------------------------------------------------------------


% Calculate threshold

MQE_mean = mean(MQE_train_1);   % Calc. mean
MQE_std  =  std(MQE_train_1);   % Calc. std
MQE_threshold = MQE_mean + 3 * MQE_std;

for ii = 1:N_test 
    if MQE_test_1(ii,1) > MQE_threshold
        MQE_predict(ii) = 1;
    else
        MQE_predict(ii) = 0;
    end
end

% ------------------------------------------------
% Plot Results

plot_threshold(:) = zeros(N_test+N_train,1);
plot_threshold(:) = MQE_threshold;
XX = (1:N_test+N_train);
MQE = [MQE_train_1; MQE_test_1];



% ---

plot_threshold(:) = zeros(N_test+N_train,1);
plot_threshold(:) = MQE_threshold;
XX0 = (1:N_test+N_train);
XX1 = (1:N_train);
XX2 = (N_train+1:N_train+N_test);
YY_train   = MQE_train_1;
YY_faulty = MQE_test_1'.*a'; 
YY_healthy  = MQE_test_1'.*c';
MQE = [MQE_train_1; MQE_test_1];

figure;
semilogy(XX1,YY_train,"ob",XX2,YY_healthy,"og",XX2,YY_faulty,"or",XX0,plot_threshold,'r');
xlim([1,N_train+N_test]);
sgtitle("Fig. Result: SOM-MQE");



%% Output prediction

% To Be Done

%%

% confusion matrix




A = a';
B = MQE_predict;
C = tt_predict;
D = SPE_predict;
E = tt_predict_a;

%======================================
figure;
truth = categorical( A );
predict = categorical( B );
plotconfusion(truth,predict)
title('SOM-MQE')

%======================================
figure;
truth = categorical( A );
predict = categorical( C );
plotconfusion(truth,predict)
title('T2(old)')

%=======================================
figure;
truth = categorical( A );
predict = categorical( D );
plotconfusion(truth,predict)
title('SPE')

%=======================================
figure;
truth = categorical( A );
predict = categorical( E );
plotconfusion(truth,predict)
title('T2')



