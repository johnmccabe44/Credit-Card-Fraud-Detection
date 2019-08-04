%% (RE-)LOAD DATA
clc
close all
clear all

addpath(genpath('lib\'));
load('data\creditcard.mat');
load('data\mySamples.mat');
load('data\myresults.mat');
load('data\testData.mat');

%% 

% Set up some sampling specific variables
SampleColors = zeros(7,3);
SampleShortDescriptions = {'Base model','5% under','10% under','50% under', ...
                            '5% over','10% over','50% over'};
                        
% Set up separate colors for each group

SampleColors(1,:) = [228/255 26/255 28/255];
SampleColors(2,:) = [247/255 129/255 191/255];
SampleColors(3,:) = [77/255 175/255 74/255];
SampleColors(4,:) = [152/255 78/255 163/255];
SampleColors(5,:) = [255/255 127/255 0/255];             
SampleColors(6,:) = [166/255 86/255 40/255];           
SampleColors(7,:) = [55/255 126/255 184/255];
SampleColors(8,:) = [153/255 153/255 153/255];

% set up xticks
xticknb=[];xtickall=[];xtickall=[];
for i = 1:14
    if i < 8
        xticknb(i) = (2.5 + (i-1)*4);
        xtickall(i) = (2.5 + (i-1)*4);
    else
        xtickrf(i-7) = (6.5 + (i-8)*12);
        xtickall(i) = (6.5 + (i-1)*12);
    end
end

%% NAIVE BAYES INITIAL INVESTIGATIONS
% based on in sample error before taking forward to final analysis
figure('Position', [0 0 1720 1000]);

% local variables for ease of plotting
models = zeros(28,1);
insLoss = zeros(28,1);
trainTime = zeros(28,1);
grp = zeros(28,1);
mrk = '.';sz=30;

% Load up data for NB into local variables for ease of plotting
for i = 1:28
    NBDescriptions = myResults.m_results(i).m_description;
    grp(i) = myResults.m_results(i).m_sampleidx;
    models(i) = i;
    insLoss(i) = myResults.m_results(i).m_insLoss;
    trainTime(i) = myResults.m_results(i).m_trainTime;
end

% run analysis on Naive Bayes
subplot(2,2,1);
plt = gscatter(models,insLoss,grp,SampleColors,mrk,sz,'off');
title("Naive Bayes: Sampling Method By In sample loss");
xticks(xticknb)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sampling Type");
ylabel("In Sample Loss");
legend(plt,SampleShortDescriptions,'location','northwest');
% RESULTS: This clearly shows that the best sampling method is No Sampling!

% Another key metric identified is the trining time; plot by model to
% determine if the best performing models take prohibitely long to train
subplot(2,2,2);
plt = gscatter(models,trainTime,grp,SampleColors,mrk,sz,'off');
title("Naive Bayes: Sample By Training Time");
xticks(xticknb)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sample");
ylabel("Time (secs)");
legend(plt,SampleShortDescriptions,'location','northwest');

% RESULTS: As epected the training time is higher for samples with larger sizes

% NAIVE BAYES HYPERPARAMETER INVESTIGATION
% 1. 
subplot(2,2,3);
mrk='';sz='';
for i = 1:28
    if mod(i,4) == 1 || mod(i,4) == 2
        mrk = 'o';
    else
        mrk = '*';
    end
    scatter(models(i), insLoss(i), [], ...
    SampleColors(myResults.m_results(i).m_sampleidx, :),mrk)     
    hold on;
end

%gscatter(models,insLoss,grp,SampleColors,mrk,sz,'off');
hold on;
title("Naive Bayes: All & Reduced Features By In Sample Loss");
xticks(xticknb)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sample");
ylabel("In Sample Loss");
h = zeros(2,1); h(1) = scatter(NaN,NaN,[],[0 0 0],'o');h(2) = scatter(NaN,NaN,[],[0 0 0],'*');
legend(h, {'All Features','Reduced Features'},'location','northwest');

%RESULTS: Shows that reduced features peforms better than All Features

subplot(2,2,4);
mrk='';sz=[];
for i = 1:28
    if mod(i,2) == 1
        mrk = 'o';
    else
        mrk = '*';
    end
    scatter(models(i), insLoss(i), [], ...
    SampleColors(myResults.m_results(i).m_sampleidx, :),mrk)     
    hold on;
    
end
%gscatter(models,insLoss,grp,SampleColors,mrk,sz,'off');
hold on;
title("Naive Bayes: Normal & Kernal (with Epanechnikov smoothing) By In Sample Loss");
xticks(xticknb)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sample");
ylabel("In Sample Loss");
h = zeros(2,1); h(1) = scatter(NaN,NaN,[],[0 0 0],'o');h(2) = scatter(NaN,NaN,[],[0 0 0],'*');
legend(h, {'Normal','Kernal(Epanechnikov)'},'location','northwest');

%RESULTS: Shows that using Epanechnikov smoothing gives better results than
% normal

figure(13);
mrk='';sz=[];
for i = 1:28
    if mod(i,2) == 1
        mrk = 'o';
    else
        mrk = '*';
    end
    
    if mod(i,4) == 1 || mod(i,4) == 2
        c = 'r';
    else
        c = 'b';
    end

    if mrk == 'o'
        scatter(models(i), insLoss(i), 100, c,mrk, 'filled');
    else
         scatter(models(i), insLoss(i), 100, c,mrk);
    end

    hold on;
    
end
%gscatter(models,insLoss,grp,SampleColors,mrk,sz,'off');
hold on;
title("Naive Bayes: Hyper-parmaters Overview By In Sample Loss");
xticks(xticknb)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sampling Type");
ylabel("In Sample Loss");
ylim([0 0.15]);
h = zeros(4,1);
h(1) = scatter(NaN,NaN,30,'r','o','filled');
h(2) = scatter(NaN,NaN,30,'b','o','filled');
h(3) = scatter(NaN,NaN,40,[0 0 0],'o','filled');
h(4) = scatter(NaN,NaN,40,[0 0 0],'*');

legend(h, {'All features','Reduced features','Normal','Kernal(Epanechnikov)'} ...
    ,'location','northwest');


%% RANDOM FORESTS INITIAL INVESTIGATIONS
% Load up data into 
% local variables for ease of plotting
figure('Position', [0 0 1720 1000]);
subplot(2,2,1);
trainTime = [];models=[];
for i = 29:112
    j = i - 28;
    % plot 
    plt(j) = plot(myResults.m_results(i).m_insLoss, ...
            'Color',SampleColors(myResults.m_results(i).m_sampleidx,:));
    hold on;
end
title('Classification Error from Random Forest Trees'); 
xlabel("Number of Trees");
ylabel("In Sample Loss");
legend([plt(1) plt(13) plt(25) plt(37) plt(49) plt(61) plt(73)],...
    SampleShortDescriptions ,'location','northwest');

% for legend

% % difficult to see which models perform best - zoom in
subplot(2,2,2);
for i = 29:112
    j = i - 28;
    plt(j) = plot(myResults.m_results(i).m_insLoss, ...
            'Color',SampleColors(round(j/12)+1,:));%myResults.m_results(i).m_sampleidx;
    hold on;
end
title('Classification Error from Random Forest Trees Zoomed to 0-0.05'); 
xlabel("Number of Trees");
ylabel("In Sample Loss");
ylim([0 0.05]);
legend([plt(1) plt(13) plt(25) plt(37) plt(49) plt(61) plt(73)], ...
    SampleShortDescriptions ,'location','northwest'); 

%
subplot(2,2,3);
grp=[];models=[];trainTime=[];
for i = 29:112
    j = i - 28;
    % store train time for later
    models(j) = j;
    trainTime(j) = myResults.m_results(i).m_trainTime;
    grp(j) = myResults.m_results(i).m_sampleidx;
    
    
    plt(j) = plot(myResults.m_results(i).m_insLoss, ...
            'Color',SampleColors(myResults.m_results(i).m_sampleidx,:));
    hold on;
    
    
end
title('Classification Error from Random Forest Trees Zoomed to 0-0.001');
xlabel("Number of Trees");
ylabel("In Sample Loss");
ylim([0 0.01]);
legend([plt(1) plt(13) plt(25) plt(37) plt(49) plt(61) plt(73)],...
    SampleShortDescriptions ,'location','northwest'); 


% Now plot time of training
subplot(2,2,4);
mrk = '.';sz=30;
plt = gscatter(models,trainTime,grp,SampleColors,mrk,sz,'off');
title("Random Forest: Sample By Training Time");
xticks(xtickrf)
xticklabels(SampleShortDescriptions)
ylim([0 0.15]);
xlabel("Model by Sample");
ylabel("Time (secs)");
legend(plt,SampleShortDescriptions,'location','northwest');


% RESULTS show that there are 4 sampling types that perform best;
% NoSampling (1-12): Over sampling 5% (49-60): Over sampling
% 10% (61-72) and Over sampling 50% (73-84). 

%% RANDOM FOREST HYPERPARAMETER ANALYSIS
% Reminder of hyper parameters: 
% hyper1Descriptions = {'Split criterian:gdi', 'Split criterian:deviance'}
% hyper1Parameters = {'gdi','deviance'};
% -
% hyper2Descriptions = {'MinLeafSize:1','MinLeafSize:5','MinLeafSize:10'};
% hyper2Parameters = [1 5 10];
fig = figure(3);
subplot(2,2,1);
insLoss=[];
for i = 29:112
    % deal with the shift due to NB
    j = i - 28;

    % local variables for ease of plotting
    grp(j) = myResults.m_results(i).m_sampleidx;
    
    % get the training labels from the mysample
    trnLabels = mySamples.m_samples(myResults.m_results(i).m_sampleidx).m_trnClasses;
    
    % insLoss
    insLoss(j) = myResults.m_results(i).m_insLoss(25,1);
    
    % set the group
    grp(j) = myResults.m_results(i).m_sampleidx;
    
    % set the models
    models(j) = j;

    % Set the marker based on whether its all features or reduced
    if mod(j-1,12) < 6
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'o','filled')
    else
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'*')
    end
     
    hold on;
end
%gscatter(models,oosLoss,grp,SampleColors,mrk,sz,'off');
title("Random Forest: Feature Selection By In Sample Loss");
xticks(xtickrf)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sample");
ylabel("In Sample Loss");
ylim([0 0.02]);
h = zeros(2,1);
h(1) = scatter(NaN,NaN,[],[0 0 0],'o','filled');
h(2) = scatter(NaN,NaN,[],[0 0 0],'*');
legend(h, {'All Features','Reduced Features'},'location','northeast');

% RESULTS NOT CLEAR WHETHER THE REDUDED FESATURES HAVE SIGNIFICANT IMPACT
subplot(2,2,2);
for i = 29:112
    % deal with the shift due to NB
    j = i - 28;
    
    % Set the marker based on whether its split type
    if mod(j-1,6) < 3
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'o','filled')
    else
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'*')
    end  
    hold on;
    
 end

%gscatter(models,oosLoss,grp,SampleColors,mrk,sz,'off');
title("Random Forest: Split Criterian By In Sample Loss");
xticks(xtickrf)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sample");
ylabel("In Sample Loss");
ylim([0 0.005]);
h = zeros(2,1);
h(1) = scatter(NaN,NaN,[],[0 0 0],'o');
h(2) = scatter(NaN,NaN,[],[0 0 0],'*');
legend(h, {'Split: gdi','Split: deviance'},'location','northeast');

subplot(2,2,3);
for i = 29:112
    % deal with the shift due to NB
    j = i - 28;
    % Set the marker based on whether the leaf size group its in
    if mod(j,3) == 1 % min leaf 1
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'o','filled')
    elseif  mod(j,3) == 2 % min leaf 5
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'*')
    else
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'x')
    end   
    hold on;
    
end
%gscatter(models,oosLoss,grp,SampleColors,mrk,sz,'off');
title("Random Forest: Min Leaf Size By In Sample Loss");
xticks(xtickrf)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sample");
ylabel("In Sample Loss");
ylim([0 0.005]);
h = zeros(3,1);
h(1) = scatter(NaN,NaN,[],[0 0 0],'o');
h(2) = scatter(NaN,NaN,[],[0 0 0],'*');
h(3) = scatter(NaN,NaN,[],[0 0 0],'x');
legend(h, {'Min Leaf: 1','Min Leaf: 5','Min Leaf: 10'},'location','northeast');

subplot(2,2,4);
for i = 29:112
    % deal with the shift due to NB
    j = i - 28;
    
    % now set the marker based on whether it is the target group or not
    if mod(j,12) == 8
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'o','filled')
    else
        scatter(models(j), insLoss(j), [], ...
        SampleColors(myResults.m_results(i).m_sampleidx, :),'*')
    end   
    hold on;

end
%gscatter(models,oosLoss,grp,SampleColors,mrk,sz,'off');
title("Random Forest: 'All Features, deviance, Min5' By In Sample Loss");
xticks(xtickrf)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sample");
ylabel("In Sample Loss");
ylim([0 0.005]);
h = zeros(2,1);
h(1)=scatter(NaN,NaN,[],[0 0 0],'*');
h(2)=scatter(NaN,NaN,[],[0 0 0],'o');
legend(h, {'all features|split:deviance|minleaf:5','everything else'},'location','northeast');

% RESULTS: This indicates for over-sampling the best combination of
% hyperparameters is All Features/ deviance/ Min Leaf Size 5 ... unlike
% Naive Bayes the Sampling method has less of an impact on performance than
% the hyper prarameters


figure(20);
for i = 29:112
    % deal with the shift due to NB
    j = i - 28;

    if mod(j,3) == 1 % min leaf 1
        mrk = 'o';
    elseif  mod(j,3) == 2 % min leaf 5
        mrk = 'd';
    else
        mrk = 's';
    end   
    
    % Set the marker based on whether its all features or reduced
    if mod(j-1,12) < 6
        c='r'
    else
        c='b';
    end
    
    if mod(j-1,6) < 3
        scatter(models(j), insLoss(j), 100, ...
                    c, mrk, 'filled')        
    else
        scatter(models(j), insLoss(j), 100, ...
                    c, mrk)        
    end  
        
    % now set the marker based on whether it is the target group or not
    hold on;

end
%gscatter(models,oosLoss,grp,SampleColors,mrk,sz,'off');
title("Random Forest: Overview Hyperparameters");
xticks(xtickrf)
xticklabels(SampleShortDescriptions)
xlabel("Model by Sample");
ylabel("In Sample Loss");
ylim([0 0.0035]);
h = zeros(7,1);
h(1) = scatter(NaN,NaN,50,'r','o','filled');
h(2) = scatter(NaN,NaN,50,'b','o','filled');
h(3) = scatter(NaN,NaN,50,[0 0 0],'o','filled');
h(4) = scatter(NaN,NaN,50,[0 0 0],'o');
h(5) = scatter(NaN,NaN,50,[0 0 0],'o','filled');
h(6) = scatter(NaN,NaN,50,[0 0 0],'d','filled');
h(7) = scatter(NaN,NaN,50,[0 0 0],'s','filled');
legend(h, {'All features','Reduced features','Split: gdi','Split: deviance','Min Leaf: 1','Min Leaf: 5','Min Leaf: 10'},'location','northwest');

%% FINAL RESULTS
%
figure('Position', [0 0 1720 1000]);
modelsToTest = [1:4 29:40 77:112];
grp=[];grpC=zeros(2,3);insLoss=[];
for i=1:112      
    if i< 29
        grp(i)=1; grpC(1,:) = [1 0 0];
    else
        grp(i)=2; grpC(2,:) = [0 0 1];
    end     
    models(i) = i;
    insLoss(i) = myResults.m_results(i).m_oosLoss;
end

gscatter(models,insLoss,grp,grpC,'.',30,'off')
hold on;
title("Classifer Comparison Based on Out Of Sample Loss");
ylabel("Out Of Bag Error");
ylim([0 0.01]);
xlabel("Trained Models");
h = zeros(2,1);h(1) = plot(NaN,NaN,'red');h(2) = plot(NaN,NaN,'blue');
legend(h,{"Naive Bayes", "Random Forests"});

%% AUROC & PRROC to illustrate why AUROC is not a suitable model
figure(5);
f1 = zeros(112,1);
modelsToTest = [6 11 13];
modelsToTest = 1:112;
auc = [];
for i=1:112
    
    % Get the f score
    f1(i) = myResults.m_results(i).CalculateF1Score();
    
    % Now we can go with the roc curve
    if ismember(i,modelsToTest)
        
        labels = testData{:,end};
        scores = myResults.m_results(i).m_cost(:,2);
        
        [X,Y,T, AUC] = perfcurve(labels,scores,1);
        auc(i) = mean(AUC);
        plot(X,Y);
        hold on;
    end
    
end

%% Conclusions
% Pretty conclusive that RF is better in this use case then Naive Bayes ...
% but just how in monetary terms is RF than NB ... load up the financial
% cost into an array and pull out the lowest loss for NB and the lowest
% loss for RF
moneyCost = zeros(112,2);savings = zeros(112,1);
for i = 1:112
    if i < 29
        moneyCost(i,:) = [1 myResults.m_results(i).CalculateFinancialCost()];
    else
        moneyCost(i,:) = [2 myResults.m_results(i).CalculateFinancialCost()];
    end
    
    savings(i,1) = (sum(testData{:,end}) * 122.21)-moneyCost(i,2);
end

[mn1 idx1] = min(moneyCost(moneyCost(:,1) == 1,2));
fprintf("\nTotal cost of Fraud with no classifiers:£%.2f\n", ...
                                sum(testData{:,end}) * 122.21);

fprintf("\nLowest cost of Naive Bayes: £%.2f\n--->Model Description: %s\n--->Savings: £%.2f\n", ...
                                mn1, ...
                                myResults.m_results(idx1).m_description, ...
                                (sum(testData{:,end}) * 122.21)-mn1);
                            
[mn2, idx2] = min(moneyCost(moneyCost(:,1) == 2,2));
fprintf("\nLowest cost of Random Forest: £%.2f\n--->Model Description: %s\n--->Savings: £%.2f\n", ...
                                mn2, ...
                                myResults.m_results(idx2+28).m_description, ...
                                (sum(testData{:,end}) * 122.21)-mn2);
                            
fprintf("\nSaving of £%.2f using Random Forest\n",mn1-mn2);                           

%% 
ModelDescriptions = {"NB,All,Normal","NB,All,Kernal","NB,Reduced,Normal","NB,Reduced,Kernal","RF,All,gdi,1","RF,All,gdi,5","RF,All,gdi,10","RF,All,Split criterian:deviance,1","RF,All,Split criterian:deviance,5","RF,All,Split criterian:deviance,10","RF,Reduced,gdi,1","RF,Reduced,gdi,5","RF,Reduced,gdi,10","RF,Reduced,Split criterian:deviance,1","RF,Reduced,Split criterian:deviance,5","RF,Reduced,Split criterian:deviance,10"};
figure(6);
subplot(1,2,1);
cdata = reshape(auc,[7 16]);
yvalues = SampleShortDescriptions;
xvalues = ModelDescriptions;
heatmap(xvalues, yvalues, cdata);
title("Heatmap of AUC for each model");

subplot(1,2,2);
cdata = reshape(savings(:,1),[7 16]);
yvalues = SampleShortDescriptions;
xvalues = ModelDescriptions;
h2 = heatmap(xvalues, yvalues, cdata);
caxis([0 7000]);
title("Heatmap of Financial Cost for each model");
