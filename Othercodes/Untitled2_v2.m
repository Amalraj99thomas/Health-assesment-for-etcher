%% Initilization
clc;
clear all;
close all;

%% Data loading
% Baseline Data Set 1
load("D:\Uni Stuff\Spring 21\Industrial AI\Project\Project1_Etcher\Training_DataSet\Baseline1DataSet.mat");
% % Baseline Data Set 2
% load("D:\Uni Stuff\Spring 21\Industrial AI\Project\Project1_Etcher\Training_DataSet\Baseline2DataSet.mat");
% % Baseline Data Set 3
% load("D:\Uni Stuff\Spring 21\Industrial AI\Project\Project1_Etcher\Training_DataSet\Baseline3DataSet.mat");
% % Testing Data Set 1
% load("D:\Uni Stuff\Spring 21\Industrial AI\Project\Project1_Etcher\Testing_DataSet\TestCase1DataSet.mat");
% % Testing Data Set 2
% load("D:\Uni Stuff\Spring 21\Industrial AI\Project\Project1_Etcher\Testing_DataSet\TestCase2DataSet.mat");
% % Testing Data Set 3
% load("D:\Uni Stuff\Spring 21\Industrial AI\Project\Project1_Etcher\Testing_DataSet\TestCase3DataSet.mat");

%% Random shit
for i=1:25
    a = Baseline1Run{i,1}.Data;
    mean1(i,:) = mean(a);
    var1(i,:) = var(a);
    skewness1(i,:) = skewness(a);
    kurtosis1(i,:) = kurtosis(a);
    rms1(i,:) = rms(a);
    peak2peak1(i,:) = peak2peak(a);
    std1(i,:) = std(a);
    max1(i,:) = max(a);
    min1(i,:) = min(a);
    feature1(i,:) = [mean1(i,:),var1(i,:),skewness1(i,:),kurtosis1(i,:),rms1(i,:),peak2peak1(i,:),std1(i,:),max1(i,:),min1(i,:)];
%     b = Baseline2Run{i,1}.Data;
%     mean2(i,:) = mean(b);
%     var2(i,:) = var(b);
%     skewness2(i,:) = skewness(b);
%     kurtosis2(i,:) = kurtosis(b);
%     rms2(i,:) = rms(b);
%     peak2peak2(i,:) = peak2peak(b);
%     std2(i,:) = std(b);
%     c = Baseline3Run{i,1}.Data;
%     mean3(i,:) = mean(c);
%     var3(i,:) = var(c);
%     skewness3(i,:) = skewness(c);
%     kurtosis3(i,:) = kurtosis(c);
%     rms3(i,:) = rms(c);
%     peak2peak3(i,:) = peak2peak(c);
%     std3(i,:) = std(c);
end


% [f1,ps]=mapstd(feature1');
% f1(isnan(f1))=0;
% [idx c sumd]= kmeans(f1',7); 
% [coeff,score,latent,tsquared,explained,mu] = pca(f1');
% figure();
% biplot(coeff(:,3:4),'scores',score(:,3:4));
% xlabel('PC1');
% ylabel('PC2');
% figure();
% pareto(explained);
% xlabel('Principal Component');
% ylabel('Variance Explained (%)');
% redfeature1 = f1'*coeff;
% pc1=score(:,3);
% pc2=score(:,4);
% figure();
% gscatter(pc1,pc2,idx);

BCI3_Flow = [mean1(:,3),var1(:,3),skewness1(:,3),kurtosis1(:,3),rms1(:,3),peak2peak1(:,3),std1(:,3),max1(:,3),min1(:,3)];
figure();
subplot(3,3,1);
plot(1:25,BCI3_Flow(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,BCI3_Flow(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,BCI3_Flow(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,BCI3_Flow(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,BCI3_Flow(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,BCI3_Flow(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,BCI3_Flow(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,BCI3_Flow(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,BCI3_Flow(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of BCI3 Flow (Variable 3)');

CI2_Flow = [mean1(:,4),var1(:,4),skewness1(:,4),kurtosis1(:,4),rms1(:,4),peak2peak1(:,4),std1(:,4),max1(:,4),min1(:,4)];
figure();
subplot(3,3,1);
plot(1:25,CI2_Flow(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,CI2_Flow(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,CI2_Flow(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,CI2_Flow(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,CI2_Flow(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,CI2_Flow(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,CI2_Flow(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,CI2_Flow(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,CI2_Flow(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of CI2 Flow (Variable 4)');

RF_Btm_Pwr = [mean1(:,5),var1(:,5),skewness1(:,5),kurtosis1(:,5),rms1(:,5),peak2peak1(:,5),std1(:,5),max1(:,5),min1(:,5)];
figure();
subplot(3,3,1);
plot(1:25,RF_Btm_Pwr(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,RF_Btm_Pwr(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,RF_Btm_Pwr(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,RF_Btm_Pwr(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,RF_Btm_Pwr(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,RF_Btm_Pwr(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,RF_Btm_Pwr(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,RF_Btm_Pwr(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,RF_Btm_Pwr(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of RF Btm Pwr (Variable 5)');

RF_Btm_Rfl_Pwr = [mean1(:,6),var1(:,6),skewness1(:,6),kurtosis1(:,6),rms1(:,6),peak2peak1(:,6),std1(:,6),max1(:,6),min1(:,6)];
figure();
subplot(3,3,1);
plot(1:25,RF_Btm_Rfl_Pwr(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,RF_Btm_Rfl_Pwr(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,RF_Btm_Rfl_Pwr(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,RF_Btm_Rfl_Pwr(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,RF_Btm_Rfl_Pwr(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,RF_Btm_Rfl_Pwr(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,RF_Btm_Rfl_Pwr(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,RF_Btm_Rfl_Pwr(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,RF_Btm_Rfl_Pwr(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of RF Btm Rfl Pwr (Variable 6)');

Endpt_A = [mean1(:,7),var1(:,7),skewness1(:,7),kurtosis1(:,7),rms1(:,7),peak2peak1(:,7),std1(:,7),max1(:,7),min1(:,7)];
figure();
subplot(3,3,1);
plot(1:25,Endpt_A(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,Endpt_A(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,Endpt_A(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,Endpt_A(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,Endpt_A(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,Endpt_A(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,Endpt_A(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,Endpt_A(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,Endpt_A(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of Endpt A (Variable 7)');

He_Press = [mean1(:,8),var1(:,8),skewness1(:,8),kurtosis1(:,8),rms1(:,8),peak2peak1(:,8),std1(:,8),max1(:,8),min1(:,8)];
figure();
subplot(3,3,1);
plot(1:25,He_Press(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,He_Press(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,He_Press(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,He_Press(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,He_Press(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,He_Press(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,He_Press(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,He_Press(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,He_Press(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of He Press (Variable 8)');

Pressure = [mean1(:,9),var1(:,9),skewness1(:,9),kurtosis1(:,9),rms1(:,9),peak2peak1(:,9),std1(:,9),max1(:,9),min1(:,9)];
figure();
subplot(3,3,1);
plot(1:25,Pressure(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,Pressure(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,Pressure(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,Pressure(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,Pressure(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,Pressure(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,Pressure(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,Pressure(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,Pressure(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of Pressure (Variable 9)');

RF_Tuner = [mean1(:,10),var1(:,10),skewness1(:,10),kurtosis1(:,10),rms1(:,10),peak2peak1(:,10),std1(:,10),max1(:,10),min1(:,10)];
figure();
subplot(3,3,1);
plot(1:25,RF_Tuner(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,RF_Tuner(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,RF_Tuner(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,RF_Tuner(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,RF_Tuner(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,RF_Tuner(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,RF_Tuner(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,RF_Tuner(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,RF_Tuner(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of RF Tuner (Variable 10)');

RF_Load = [mean1(:,11),var1(:,11),skewness1(:,11),kurtosis1(:,11),rms1(:,11),peak2peak1(:,11),std1(:,11),max1(:,11),min1(:,11)];
figure();
subplot(3,3,1);
plot(1:25,RF_Load(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,RF_Load(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,RF_Load(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,RF_Load(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,RF_Load(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,RF_Load(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,RF_Load(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,RF_Load(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,RF_Load(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of RF Load (Variable 11)');

RF_Phase_Err = [mean1(:,12),var1(:,12),skewness1(:,12),kurtosis1(:,12),rms1(:,12),peak2peak1(:,12),std1(:,12),max1(:,12),min1(:,12)];
figure();
subplot(3,3,1);
plot(1:25,RF_Phase_Err(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,RF_Phase_Err(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,RF_Phase_Err(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,RF_Phase_Err(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,RF_Phase_Err(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,RF_Phase_Err(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,RF_Phase_Err(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,RF_Phase_Err(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,RF_Phase_Err(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of RF Phase Err (Variable 12)');

RF_Pwr = [mean1(:,13),var1(:,13),skewness1(:,13),kurtosis1(:,13),rms1(:,13),peak2peak1(:,13),std1(:,13),max1(:,13),min1(:,13)];
figure();
subplot(3,3,1);
plot(1:25,RF_Pwr(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,RF_Pwr(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,RF_Pwr(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,RF_Pwr(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,RF_Pwr(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,RF_Pwr(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,RF_Pwr(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,RF_Pwr(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,RF_Pwr(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of RF Pwr (Variable 13)');

RF_Impedance = [mean1(:,14),var1(:,14),skewness1(:,14),kurtosis1(:,14),rms1(:,14),peak2peak1(:,14),std1(:,14),max1(:,14),min1(:,14)];
figure();
subplot(3,3,1);
plot(1:25,RF_Impedance(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,RF_Impedance(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,RF_Impedance(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,RF_Impedance(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,RF_Impedance(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,RF_Impedance(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,RF_Impedance(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,RF_Impedance(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,RF_Impedance(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of RF Impedance (Variable 14)');

TCP_Tuner = [mean1(:,15),var1(:,15),skewness1(:,15),kurtosis1(:,15),rms1(:,15),peak2peak1(:,15),std1(:,15),max1(:,15),min1(:,15)];
figure();
subplot(3,3,1);
plot(1:25,TCP_Tuner(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,TCP_Tuner(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,TCP_Tuner(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,TCP_Tuner(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,TCP_Tuner(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,TCP_Tuner(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,TCP_Tuner(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,TCP_Tuner(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,TCP_Tuner(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of TCP Tuner (Variable 15)');

TCP_Phase_Err = [mean1(:,16),var1(:,16),skewness1(:,16),kurtosis1(:,16),rms1(:,16),peak2peak1(:,16),std1(:,16),max1(:,16),min1(:,16)];
figure();
subplot(3,3,1);
plot(1:25,TCP_Phase_Err(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,TCP_Phase_Err(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,TCP_Phase_Err(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,TCP_Phase_Err(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,TCP_Phase_Err(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,TCP_Phase_Err(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,TCP_Phase_Err(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,TCP_Phase_Err(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,TCP_Phase_Err(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of TCP Phase Err (Variable 16)');

TCP_Impedance = [mean1(:,17),var1(:,17),skewness1(:,17),kurtosis1(:,17),rms1(:,17),peak2peak1(:,17),std1(:,17),max1(:,17),min1(:,17)];
figure();
subplot(3,3,1);
plot(1:25,TCP_Impedance(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,TCP_Impedance(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,TCP_Impedance(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,TCP_Impedance(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,TCP_Impedance(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,TCP_Impedance(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,TCP_Impedance(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,TCP_Impedance(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,TCP_Impedance(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of TCP Impedance (Variable 17)');

TCP_Top_Pwr = [mean1(:,18),var1(:,18),skewness1(:,18),kurtosis1(:,18),rms1(:,18),peak2peak1(:,18),std1(:,18),max1(:,18),min1(:,18)];
figure();
subplot(3,3,1);
plot(1:25,TCP_Top_Pwr(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,TCP_Top_Pwr(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,TCP_Top_Pwr(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,TCP_Top_Pwr(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,TCP_Top_Pwr(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,TCP_Top_Pwr(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,TCP_Top_Pwr(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,TCP_Top_Pwr(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,TCP_Top_Pwr(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of TCP Top Pwr (Variable 18)');

TCP_Rfl_Pwr = [mean1(:,19),var1(:,19),skewness1(:,19),kurtosis1(:,19),rms1(:,19),peak2peak1(:,19),std1(:,19),max1(:,19),min1(:,19)];
figure();
subplot(3,3,1);
plot(1:25,TCP_Rfl_Pwr(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,TCP_Rfl_Pwr(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,TCP_Rfl_Pwr(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,TCP_Rfl_Pwr(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,TCP_Rfl_Pwr(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,TCP_Rfl_Pwr(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,TCP_Rfl_Pwr(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,TCP_Rfl_Pwr(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,TCP_Rfl_Pwr(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of TCP Rfl Pwr (Variable 19)');

TCP_Load = [mean1(:,20),var1(:,20),skewness1(:,20),kurtosis1(:,20),rms1(:,20),peak2peak1(:,20),std1(:,20),max1(:,20),min1(:,20)];
figure();
subplot(3,3,1);
plot(1:25,TCP_Load(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,TCP_Load(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,TCP_Load(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,TCP_Load(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,TCP_Load(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,TCP_Load(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,TCP_Load(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,TCP_Load(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,TCP_Load(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of TCP Load (Variable 20)');

Vat_Valve = [mean1(:,21),var1(:,21),skewness1(:,21),kurtosis1(:,21),rms1(:,21),peak2peak1(:,21),std1(:,21),max1(:,21),min1(:,21)];
figure();
subplot(3,3,1);
plot(1:25,Vat_Valve(:,1));
xlabel('Number of Runs');
ylabel('Mean');

subplot(3,3,2);
plot(1:25,Vat_Valve(:,2));
xlabel('Number of Runs');
ylabel('Variance');

subplot(3,3,3);
plot(1:25,Vat_Valve(:,3));
xlabel('Number of Runs');
ylabel('Skewness');

subplot(3,3,4);
plot(1:25,Vat_Valve(:,4));
xlabel('Number of Runs');
ylabel('Kurtosis');

subplot(3,3,5);
plot(1:25,Vat_Valve(:,5));
xlabel('Number of Runs');
ylabel('RMS');

subplot(3,3,6);
plot(1:25,Vat_Valve(:,6));
xlabel('Number of Runs');
ylabel('Peak2Peak');

subplot(3,3,7);
plot(1:25,Vat_Valve(:,7));
xlabel('Number of Runs');
ylabel('Standard Deviation');

subplot(3,3,8);
plot(1:25,Vat_Valve(:,8));
xlabel('Number of Runs');
ylabel('Max');

subplot(3,3,9);
plot(1:25,Vat_Valve(:,9));
xlabel('Number of Runs');
ylabel('Min');
sgtitle('Statistical features of Vat Valve (Variable 21)');

label_mean = strcat('Mean of'," ", VarName);
label_var = strcat('Variance of'," ", VarName);
label_skewness = strcat('Skewness of'," ", VarName);
label_kurtosis = strcat('Kurtosis of'," ", VarName);
label_rms = strcat('RMS of'," ", VarName);
label_peak2peak = strcat('Peak2Peak of'," ", VarName);
label_std = strcat('Standard Deviation of'," ", VarName);
label_max = strcat('Max of'," ", VarName);
label_min = strcat('Min of'," ", VarName);
feature_labels = [label_mean,label_var,label_skewness,label_kurtosis,label_rms,label_peak2peak,label_std,label_max,label_min];

% y = Baseline1Run{1,1}.Data;
% figure(1);
% subplot(5,4,1);
% plot(y(:,1),y(:,3));
% xlabel("Time"); ylabel("BCI3 Flow");
% subplot(5,4,2);
% plot(y(:,1),y(:,4));
% xlabel("Time"); ylabel("CI2 Flow");
% subplot(5,4,3);
% plot(y(:,1),y(:,5));
% xlabel("Time"); ylabel("RF Btm Pwr");
% subplot(5,4,4);
% plot(y(:,1),y(:,6));
% xlabel("Time"); ylabel("RF Btm Rfl Pwr");
% subplot(5,4,5);
% plot(y(:,1),y(:,7));
% xlabel("Time"); ylabel("Endpt A");
% subplot(5,4,6);
% plot(y(:,1),y(:,8));
% xlabel("Time"); ylabel("He Press");
% subplot(5,4,7);
% plot(y(:,1),y(:,9));
% xlabel("Time"); ylabel("Pressure");
% subplot(5,4,8);
% plot(y(:,1),y(:,10));
% xlabel("Time"); ylabel("RF Tuner");
% subplot(5,4,9);
% plot(y(:,1),y(:,11));
% xlabel("Time"); ylabel("RF Load");
% subplot(5,4,10);
% plot(y(:,1),y(:,12));
% xlabel("Time"); ylabel("RF Phase Err");
% subplot(5,4,11);
% plot(y(:,1),y(:,13));
% xlabel("Time"); ylabel("RF Pwr");
% subplot(5,4,12);
% plot(y(:,1),y(:,14));
% xlabel("Time"); ylabel("RF Impedance");
% subplot(5,4,13);
% plot(y(:,1),y(:,15));
% xlabel("Time"); ylabel("TCP Tuner");
% subplot(5,4,14);
% plot(y(:,1),y(:,16));
% xlabel("Time"); ylabel("TCP Phase Err");
% subplot(5,4,15);
% plot(y(:,1),y(:,17));
% xlabel("Time"); ylabel("TCP Impedance");
% subplot(5,4,16);
% plot(y(:,1),y(:,18));
% xlabel("Time"); ylabel("TCP Top Pwr");
% subplot(5,4,17);
% plot(y(:,1),y(:,19));
% xlabel("Time"); ylabel("TCP Rfl Pwr");
% subplot(5,4,18);
% plot(y(:,1),y(:,20));
% xlabel("Time"); ylabel("TCP Load");
% subplot(5,4,19);
% plot(y(:,1),y(:,21));
% xlabel("Time"); ylabel("Vat Valve");
% sgtitle("Raw Sensor Data for single run from Etcher 2");

% figure(2);
% subplot(3,2,1);
% plot(1:21,mean2(1,:));
% xlabel("Variables"); ylabel("Mean");
% subplot(3,2,2);
% plot(1:21,kurtosis2(1,:));
% xlabel("Variables"); ylabel("Kurtosis");
% subplot(3,2,3);
% plot(1:21,peak2peak2(1,:));
% xlabel("Variables"); ylabel("Peak to Peak");
% subplot(3,2,4);
% plot(1:21,rms2(1,:));
% xlabel("Variables"); ylabel("RMS");
% subplot(3,2,5);
% plot(1:21,skewness2(1,:));
% xlabel("Variables"); ylabel("Skewness");
% subplot(3,2,6);
% plot(1:21,var2(1,:));
% xlabel("Variables"); ylabel("Variance");
% sgtitle("Statistical Data of the variables for a single wafer");

% figure(3);
% subplot(5,4,1);
% plot(1:25,mean1(:,3));
% hold on
% plot(1:25,mean2(:,3),'r');
% hold on
% plot(1:25,mean3(:,3),'g');
% xlabel("Runs"); ylabel("BCI3 Flow");
% legend("Etcher 11","Etcher 2","Etcher 3");
% 
% subplot(5,4,2);
% plot(1:25,mean1(:,4));
% hold on
% plot(1:25,mean2(:,4),'r');
% hold on
% plot(1:25,mean3(:,4),'g');
% xlabel("Runs"); ylabel("CI2 Flow");
% 
% subplot(5,4,3);
% plot(1:25,mean1(:,5));
% hold on
% plot(1:25,mean2(:,5),'r');
% hold on
% plot(1:25,mean3(:,5),'g');
% xlabel("Runs"); ylabel("RF Btm Pwr");
% 
% subplot(5,4,4);
% plot(1:25,mean1(:,6));
% hold on
% plot(1:25,mean2(:,6),'r');
% hold on
% plot(1:25,mean3(:,6),'g');
% xlabel("Runs"); ylabel("RF Btm Rfl Pwr");
% 
% subplot(5,4,5);
% plot(1:25,mean1(:,7));
% hold on
% plot(1:25,mean2(:,7),'r');
% hold on
% plot(1:25,mean3(:,7),'g');
% xlabel("Runs"); ylabel("Endpt A");
% 
% subplot(5,4,6);
% plot(1:25,mean1(:,8));
% hold on
% plot(1:25,mean2(:,8),'r');
% hold on
% plot(1:25,mean3(:,8),'g');
% xlabel("Runs"); ylabel("He Press");
% 
% subplot(5,4,7);
% plot(1:25,mean1(:,9));
% hold on
% plot(1:25,mean2(:,9),'r');
% hold on
% plot(1:25,mean3(:,9),'g');
% xlabel("Runs"); ylabel("Pressure");
% 
% subplot(5,4,8);
% plot(1:25,mean1(:,10));
% hold on
% plot(1:25,mean2(:,10),'r');
% hold on
% plot(1:25,mean3(:,10),'g');
% xlabel("Runs"); ylabel("RF Tuner");
% 
% subplot(5,4,9);
% plot(1:25,mean1(:,11));
% hold on
% plot(1:25,mean2(:,11),'r');
% hold on
% plot(1:25,mean3(:,11),'g');
% xlabel("Runs"); ylabel("RF Load");
% 
% subplot(5,4,10);
% plot(1:25,mean1(:,12));
% hold on
% plot(1:25,mean2(:,12),'r');
% hold on
% plot(1:25,mean3(:,12),'g');
% xlabel("Runs"); ylabel("RF Phase Err");
% 
% subplot(5,4,11);
% plot(1:25,mean1(:,13));
% hold on
% plot(1:25,mean2(:,13),'r');
% hold on
% plot(1:25,mean3(:,13),'g');
% xlabel("Runs"); ylabel("RF Pwr");
% 
% subplot(5,4,12);
% plot(1:25,mean1(:,14));
% hold on
% plot(1:25,mean2(:,14),'r');
% hold on
% plot(1:25,mean3(:,14),'g');
% xlabel("Runs"); ylabel("RF Impedance");
% 
% subplot(5,4,13);
% plot(1:25,mean1(:,15));
% hold on
% plot(1:25,mean2(:,15),'r');
% hold on
% plot(1:25,mean3(:,15),'g');
% xlabel("Runs"); ylabel("TCP Tuner");
% 
% subplot(5,4,14);
% plot(1:25,mean1(:,16));
% hold on
% plot(1:25,mean2(:,16),'r');
% hold on
% plot(1:25,mean3(:,16),'g');
% xlabel("Runs"); ylabel("TCP Phase Err");
% 
% subplot(5,4,15);
% plot(1:25,mean1(:,17));
% hold on
% plot(1:25,mean2(:,17),'r');
% hold on
% plot(1:25,mean3(:,17),'g');
% xlabel("Runs"); ylabel("TCP Impedance");
% 
% subplot(5,4,16);
% plot(1:25,mean1(:,18));
% hold on
% plot(1:25,mean2(:,18),'r');
% hold on
% plot(1:25,mean3(:,18),'g');
% xlabel("Runs"); ylabel("TCP Top Pwr");
% 
% subplot(5,4,17);
% plot(1:25,mean1(:,19));
% hold on
% plot(1:25,mean2(:,19),'r');
% hold on
% plot(1:25,mean3(:,19),'g');
% xlabel("Runs"); ylabel("TCP Rfl Pwr");
% 
% subplot(5,4,18);
% plot(1:25,mean1(:,20));
% hold on
% plot(1:25,mean2(:,20),'r');
% hold on
% plot(1:25,mean3(:,20),'g');
% xlabel("Runs"); ylabel("TCP Load");
% 
% subplot(5,4,19);
% plot(1:25,mean1(:,21));
% hold on
% plot(1:25,mean2(:,21),'r');
% hold on
% plot(1:25,mean3(:,21),'g');
% xlabel("Runs"); ylabel("Vat Valve");
% sgtitle("Mean Data of the Sensors from Etcher 2");
% 
% figure(4);
% subplot(3,2,1);
% plot(1:25,BCI3_Flow(1:25,1));
% hold on
% plot(26:50,BCI3_Flow(26:50,1),'r');
% hold on
% plot(51:75,BCI3_Flow(51:75,1),'g');
% xlabel("Runs"); ylabel("Mean");
% 
% subplot(3,2,2);
% plot(1:25,BCI3_Flow(1:25,2));
% hold on
% plot(26:50,BCI3_Flow(26:50,2),'r');
% hold on
% plot(51:75,BCI3_Flow(51:75,2),'g');
% xlabel("Runs"); ylabel("Variance");
% 
% subplot(3,2,3);
% plot(1:25,BCI3_Flow(1:25,3));
% hold on
% plot(26:50,BCI3_Flow(26:50,3),'r');
% hold on
% plot(51:75,BCI3_Flow(51:75,3),'g');
% xlabel("Runs"); ylabel("Skewness");
% 
% subplot(3,2,3);
% plot(1:25,BCI3_Flow(1:25,3));
% hold on
% plot(26:50,BCI3_Flow(26:50,3),'r');
% hold on
% plot(51:75,BCI3_Flow(51:75,3),'g');
% xlabel("Runs"); ylabel("Skewness");
% 
% sgtitle("Features of BCI3 Flow from all 3 Etchers"); 