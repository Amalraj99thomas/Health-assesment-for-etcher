clear;
close all;
clc;
%This code is used to make the detection for Semiconductor etching
%INDUSTRIAL AI Group 9 final term project
%University of Cincinnati

%% add the path/file location
% addpath 'C:\Users\Anuraga Sankepally\OneDrive - University of Cincinnati\Desktop\UNI\Industrial AI\Final project\Project1_Etcher\Training_DataSet'
%% Load data 
load("D:\Uni Stuff\Spring 21\Industrial AI\Project\Project1_Etcher\Training_DataSet\Baseline1DataSet.mat");
%% testing dataset 
% addpath 'C:\Users\Anuraga Sankepally\OneDrive - University of Cincinnati\Desktop\UNI\Industrial AI\Final project\Project1_Etcher\Testing_DataSet'
% load TestCase1DataSet

%% Feature extraction
for i=1:25                        % load input data
x = Baseline1Run{i,1}.Data;
[numrows,numcols]=size(x);
for j=1:numrows                   %populate feature matrices
F(j,:)= x(j,:);
end 
feature1(i,:)=mean(F);
feature2(i,:)=min(F);
feature3(i,:)=std(F);
feature4(i,:)=rms(F);
feature5(i,:)=kurtosis(F);
feature6(i,:)=range(F);

end
%% normalized features
for i=1:21
F1(:,i)=(feature1(:,i)-mean(feature1(:,i)))/std(feature1(:,i));
F2(:,i)=(feature2(:,i)-mean(feature2(:,i)))/std(feature2(:,i));
F3(:,i)=(feature3(:,i)-mean(feature3(:,i)))/std(feature3(:,i));
F4(:,i)=(feature4(:,i)-mean(feature4(:,i)))/std(feature4(:,i));
F5(:,i)=(feature5(:,i)-mean(feature5(:,i)))/std(feature5(:,i));
F6(:,i)=(feature6(:,i)-mean(feature6(:,i)))/std(feature6(:,i));
end
%% healthy feature structure
Feature.mean=F1(:,3:21);
Feature.skewness=F2(:,3:21);
Feature.standarddev=F3(:,3:21);
Feature.rms=F4(:,3:21);
Feature.kurtosis=F5(:,3:21);
Feature.peaktopeak=F6(:,3:21);
%% taking only the useful features
 
F=cell2mat(struct2cell(Feature)); %FEATURE MATRIX HEALTH
F(isnan(F))=0;
[idx c sumd]= kmeans(F,6);        %kmeans cluster visualization(optional)
[coeff, score, latent, tsquared, explained, mu] = pca(F);
pc1=score(:,1);
pc2=score(:,2);
figure();
gscatter(pc1,pc2,idx);
%% biplot visualization
figure();
Var= VarName(3:21);
h = biplot(coeff(:,1:3),'Scores',score(:,1:3),'Varlabels',Var);
xlabel('PC1');
ylabel('PC2');
%% percentage variance explained
figure();
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
%% reduced feature matrix
new_feat=F*coeff(:,1:11);  % taking a min of 80% of the variance 

%% Test matrix dimension reduction
% addpath ('C:\Users\Anuraga Sankepally\OneDrive - University of Cincinnati\Desktop\UNI\Industrial AI\Final project\Project1_Etcher\Testing_DataSet\');
%% testing dataset 
load("D:\Uni Stuff\Spring 21\Industrial AI\Project\Project1_Etcher\Testing_DataSet\TestCase1DataSet.mat");

%% Feature extraction
for i=1:18                        % load input data
x2 = TestCase1Run{i,1}.Data;
[numrows,numcols]=size(x2);
for j=1:numrows                   %populate feature matrices
FT(j,:)= x2(j,:);
end 
feature1(i,:)=mean(FT);
feature2(i,:)=min(FT);
feature3(i,:)=std(FT);
feature4(i,:)=rms(FT);
feature5(i,:)=kurtosis(FT);
feature6(i,:)=range(FT);

end
%% normalized features
for i=1:21
FT1(:,i)=(feature1(:,i)-mean(feature1(:,i)))/std(feature1(:,i));
FT2(:,i)=(feature2(:,i)-mean(feature2(:,i)))/std(feature2(:,i));
FT3(:,i)=(feature3(:,i)-mean(feature3(:,i)))/std(feature3(:,i));
FT4(:,i)=(feature4(:,i)-mean(feature4(:,i)))/std(feature4(:,i));
FT5(:,i)=(feature5(:,i)-mean(feature5(:,i)))/std(feature5(:,i));
FT6(:,i)=(feature6(:,i)-mean(feature6(:,i)))/std(feature6(:,i));
end
%% healthy feature structure
TestFeature.mean=FT1(:,3:21);
TestFeature.skewness=FT2(:,3:21);
TestFeature.standarddev=FT3(:,3:21);
TestFeature.rms=FT4(:,3:21);
TestFeature.kurtosis=FT5(:,3:21);
TestFeature.peaktopeak=FT6(:,3:21);
%% taking only the useful features
 
FT=cell2mat(struct2cell(TestFeature)); %FEATURE MATRIX HEALTH
FT(isnan(FT))=0;
[idx c sumd]= kmeans(FT,6);        %kmeans cluster visualization(optional)
[coeff_T, score_T, latent, tsquared_T, explained_T, mu] = pca(FT);
pc1=score_T(:,1);
pc2=score_T(:,2);
figure();
gscatter(pc1,pc2,idx);
%% biplot visualization
figure();
%Var= VarName(3:21);
h = biplot(coeff_T(:,1:3),'Scores',score_T(:,1:3),'Varlabels',Var);
xlabel('PC1');
ylabel('PC2');
%% percentage variance explained
figure();
pareto(explained_T);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
%% reduced feature matrix
newtestfeat=FT*coeff_T(:,1:11);
%% Self organizing maps 
addpath('D:\Uni Stuff\Spring 21\Industrial AI\HW3\somtoolbox2_Mar_17_2005\somtoolbox');

sM=som_make(new_feat);

S=size(newtestfeat);
S=S(1);
for i=1:S
    qe=som_quality(sM,newtestfeat(i,:)); % calculate MQE value for each sample
    MQEt(i)=qe;
end
MQEtn=((MQEt)./(max(MQEt))); % normalize MQE
MQEtn=MQEtn';

%Plot MQE
figure();
plot(1:150,MQEtn,'-');
xlabel('Data file No.');
ylabel('MQE Value');
title('MQE Health Assessment Plot');










