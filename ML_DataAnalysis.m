%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: John McCabe and Kwun Ho Ngan
% Title: COST MINIMISATION FOR CREDIT CARD FRAUD DETECTION
%        USING NAÏVE BAYES AND RANDOM FOREST CLASSIFICATION
% Module: INM431 - Machine Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PREPARE WORKSPACE & LOAD DATA
clc;
clear all;
close all;

addpath(genpath('lib\'));

load("Data\creditcard.mat");
[numTotalTxn, numTotalAttr] = size(creditcard);

%% USER DEFINED CONSTANTS

% Constants are defined to create placeholder space for output matrix
MAX_SAMPLES = 10; % Max Number of User Input Dataset
MAX_RESULTS = 150; % Max Number of Hyperparameter Tuning Combinations

%% MONETARY COST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cost basis with reference to Bahnsen et al (2013).
% Cost of undetected fraud is assumed to be the average transaction loss of
% all known fraud cases
% Cost of dealing with detected fraud cases is assumed to take up 4 hours of
% administrative work at minimum national wage 2018.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

costFraud = mean(creditcard{creditcard{:,end} == 1, end-1});
costAdmin = 8.21*4; % UK Minimum Wage * 4 hours
costTxn = mean(creditcard{:,end-1});

%% EXPLORATORY ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data set (creditcard) is split into legitimate and fraud transaction.
% Variables are compared via histogram. A pairmatrix was useful to identify
% some trend but too small in screen space for visual analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

legit = creditcard(creditcard{:,end}==0,:);
fraud = creditcard(creditcard{:,end}==1,:);

% Basic Descriptive Statistics
tabulate(creditcard{:,end}) %Class distribution
features = creditcard{:,1:end-1};
tblStats = [array2table(min(features));
                array2table(max(features));
                array2table(round(mean(features)));
                array2table(std(features))                
                ];
            
tblStats.Properties.RowNames=["Min" "Max" "Mean" "Std"];
display(tblStats)

% QQ plot to visualise distribution
figure('Name', 'QQ plots of variables', 'Position', [0 0 1720 1000])
ipos = 1; %Position #
for i = 1:30
    subplot(5,12,ipos)
    qqplot(creditcard{:,i});
    title(["QQPlot:", char(creditcard.Properties.VariableNames(i))]);
    ipos = ipos + 1;
    subplot(5,12,ipos);
    histogram(creditcard{:,i},1000);
    title(["Hist:", char(creditcard.Properties.VariableNames(i))]);
    ipos = ipos + 1;
end


% Histogram for comparison (Sample of separation)
figure('Name', 'Histograms', 'Position', [0 0 1720 1000] )
for i = 1:size(features,2)
    
    subplot(5,6,i);
    h1 = histogram(legit{:,i});
    title(["Hist:", char(creditcard.Properties.VariableNames(i))]);
    hold on
    h2 = histogram(fraud{:,i});
    h1.Normalization = 'probability';
    h1.BinWidth = 1;
    h2.Normalization = 'probability';
    h2.BinWidth = 1;
end

%Pairplot
%figure(2)
%group = creditcard{:,31};
%gplotmatrix(creditcard{:,1:30},[],group, 'br', '', [], 'on', 'grpbars')

%% OUTLIER DETECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In addition to identifying the outliers that deviate from a normal 
% distribution from qq plot, outliers are also detected based on 1.5x 
% Inter-Quartile Range (IQR). 
%
% Outlier is not removed since most fraud transactions are classified as
% outliers from isoutlier function criterion.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tblOutlier = array2table(sum(isoutlier(features)))
tblOutlier.Properties.RowNames=["Outlier Count"];
display(tblOutlier)

%% SKEWNESS TRANSFORMATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Transformation for skewed features to reduce the effect of outliers
% and allow feature to resemble closer to a Gaussian Distribution.
% Important for Naive Bayes models.
%
% This is not used ultimately as the transformation blurs the split between
% fraud and legitimate transaction (Increased false positive - Models
% assume many legitimate cases as fraud).
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
[m,n] = size(creditcard);
creditcardTrans = zeros(m,n);
idxTrans = [];

for i=1:size(features,2)

    f = creditcard{:,i}; %f is the individual feature in the loop        
    % Feature is transformed if skewness exceeds 2x standard deviation
    if abs(skewness(f)) > 2*abs(std(f))

        idxTrans = [idxTrans i];
        if skewness(f) > 0
            nf = Transform(f, 'posLog');
        else
            nf = Transform(f, 'negLog');
        end

        creditcardTrans(:,i) = nf;                            
    else
        creditcardTrans(:,i) = f;
    end

end
fprintf("\n");

% Replot QQPlot before and after tansformastion with skew values
for i=1:5
    x = (i-1)*6;figure(20+i);ipos = 1;
    for j = 1:6        
        if ismember(x+j,idxTrans)
            % store variable name
            varName = char(creditcard.Properties.VariableNames(x+j));

            % plot the pre-trans variable
            subplot(3,4,ipos);
            v = table2array(creditcard(:,x+j));            
            qqplot(v);                    
            title(["QQPlot:", varName, "Skew:", skewness(v)]);
            ipos = ipos + 1;            

            % now plot post-trans variable
            subplot(3,4,ipos);                        
            v = creditcardTrans(:,x+j);            
            qqplot(v);

            title(["Trans. QQPlot:", varName, ...
                    "Skew:", skewness(v), ...
                    "Use Kernal:", abs(skewness(v)),",",2*abs(std(v)) ...
                    ]);

            ipos = ipos + 1;
        end
    end     
end

close all;

%% FEATURE SELECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Selection of variables are based on the minimisation of the overlapping
% area from the compared histograms of the legitimate and fraud
% transactions. These have been highlighted as the idxReducedFeatues
% A detailed review should be conducted on mutual information gain.
%
% Reviewing the QQPlots allowed us to determine which features were not
% normally distributed. These have been highlighed as the
% idxUseKernalFeatures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idxAllFeatures = 1:30;
idxReducedFeatures = [1 4 5 6 8 11 12 13 15 17 18 19];
idxUseKernalFeatures = [7 8 12 15 20 21 24 27 28];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% An example of feature engineering by converting time in second to hour of
% day. This allows new variable for prediction model. Fraud is found to be
% higher in the middle of the night.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Convert Time into Hour of Day
creditcard{:,1} = str2num(datestr(seconds(creditcard{:,1}),'HH'));

%% TRAINING/TEST SPLIT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% An example of holdout portion for out-of-sample testing. Cross-validation
% is applied in the training set for Naive Bayes models. Given the
% ensembled nature of Random Forest Model, cross-validation is deemed
% unnecessary for overfitting detection.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 20% HoldOut for Testing
trainCases = round(numTotalTxn * 0.80,0);

% set rand.stream for reproducibility
s = RandStream('mlfg6331_64');

% Random sampling for train/test split without replacement
[trainData, trainidx] = datasample(s, creditcard, trainCases, 'Replace', false);
testData = creditcard(setdiff(1:numTotalTxn,trainidx),:);
[testCases, testidx] = size(testData);

%% Training Data Resampling

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Re-sampling of data set to balance out the imbalanced creditcard dataset.
% Technique is based on undersampling and oversampling suggested in 
% Bahnsen et al (2013). 
% Adasyn function is built by Dominic Siedhoff (Matlab File Exchange)
% according to He et al (2008) on AdaSYN algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialise Samples Class
mySamples = SamplesClass(MAX_SAMPLES); 
% Counter for Samplesclass loops
iSamples = 1;


% Splitting Features & Output Class
trnFeatures = creditcard{trainidx,1:end-1};
trnClasses = creditcard{trainidx,end};

% Add to samples; increment counter
mys = SampleClass(trnFeatures, trnClasses, "Base Case");
mySamples.Add(iSamples,mys);
iSamples = iSamples+1;
        
% Under-Sampling   
legit = trainData(trainData{:,end}==0,:);
fraud = trainData(trainData{:,end}==1,:);

% Set up for loop
factors = [1 9 19];
descriptions = {'5% under sampling', '10% under sampling', '50% under sampling'};

% Create undersampled dataset and load that into sampleclass
for i = 1:3
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Random selection of datapoints (without replacement) for legitimate
    % transactions to make up the appropriate proportion of fraud transaction 
    % (e.g. 5%). Sort that according to time.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    l = datasample(s, legit, factors(i)*size(fraud,1), 'Replace', false);
    t = sortrows(vertcat(l,fraud),1);

    mySamples.Add(iSamples, SampleClass(table2array(t(:,1:end-1)), ...
                                        table2array(t(:,end)), ...
                                        descriptions(i)));
    iSamples = iSamples + 1;
end
    
% Over-Sampling (ADASYN)
adasyn_features                 = trainData{:,1:end-1};
adasyn_labels                   = trainData{:,end};
adasyn_kDensity                 = [];   %let ADASYN choose default
adasyn_kSMOTE                   = [];   %let ADASYN choose default
adasyn_featuresAreNormalized    = false;    %false lets ADASYN handle normalization
tname = trainData.Properties.VariableNames;

props = [0.05105 0.11025 1];
descriptions = {'Over sampling 5%','Over sampling 10%', 'Over sampling 50%'};

for i = 1:3
    [f, c] = ADASYN(adasyn_features, adasyn_labels, props(i), ...
                    adasyn_kDensity, adasyn_kSMOTE, ...
                    adasyn_featuresAreNormalized);

    % Concatenating newly synthetic fraud data (Minority)        
    fraudSyn =  array2table(horzcat(f,c), 'VariableNames', tname);
    trainOver = sortrows(vertcat(trainData,fraudSyn),1);

    mySamples.Add(iSamples, ...
                    SampleClass(trainOver(:,1:end-1), ...
                                trainOver(:,end), ...
                                descriptions(i)) ...
                    );

    iSamples = iSamples + 1;

end

%% CLASSIFIERS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section constructs both the Naive Bayes and Random Forest Models 
% based on different hyperparameter tuning and different input dataset 
% (e.g. under-sampled/over-sampled).
% In-sample & out of sample error calculation are stored as result for
% subsequent model selection and evaluation.
%
% WARNING: This section takes a few hours to run due to kernel estimation
% Number of Naive Bayes Models: 2*2*7 = 28
% Number of Random Forest Models: 2*2*3*7 = 84
% Total Number of Models: 112
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set Up Result Collection
myResults = ResultsClass(MAX_RESULTS);
iResults = 1;

% Naive Bayes Classification
for isx = 1:iSamples-1

    % Run each NB option for all features & reduced features
    featureDescriptions = {'All features','Reduced features'};

    for ifx = 1:2

        if ifx == 1
            incF = idxAllFeatures;                
        else
            incF = idxReducedFeatures;
        end

        hyper1Descriptions = {'Distributions:Normal','Distributions:Kernal'};
        for ih1x = 1:2
            % secondary index as reduced set of distribution
            ii = 1;
            distributionNames = {};
            for i = idxAllFeatures

                % A check on features within input dataset (All/Reduced)
                if ismember(i,incF)
                    % For ih1x = 1 (i.e. Normal Distribution), all features
                    % are assumed normal.
                    if ih1x == 1
                        distributionNames(ii) = {'normal'};
                    else
                        % For ih1x = 2 (i.e. Kernel), all features deviate
                        % from normal in q-q plot (see EDA section) will
                        % be estimated using kernel smoothing.
                        if ismember(i,idxUseKernalFeatures)
                            distributionNames(ii) = {'kernel'};
                        else
                            distributionNames(ii) = {'normal'};
                        end
                    end

                    % parms
                    ii = ii + 1;
                end
            end      
            
            f = mySamples.m_samples(isx).m_trnFeatures(:,incF);
            c = mySamples.m_samples(isx).m_trnClasses;

            t1 = tic;
            if (ih1x == 1 )
                mdl = fitcnb(f, c, ...
                            'DistributionNames', ...
                            distributionNames, ...
                            'HyperparameterOptimizationOptions', ...
                            struct('AcquisitionFunctionName',...
                            'expected-improvement-plus', ...
                            'UseParallel',true));
                
            else
                mdl = fitcnb(f, c, ...
                            'DistributionNames', ...
                            distributionNames, ...
                            'Kernel','epanechnikov', ...
                            'HyperparameterOptimizationOptions', ...
                            struct('AcquisitionFunctionName',...
                            'expected-improvement-plus', ...
                            'UseParallel',true));
            end
            tTrain = toc(t1);
            
            % Set up the description
            descrip = strcat("Classifier:NB,", mySamples.m_samples(isx).m_description, ",", ...
                            featureDescriptions(ifx),",",...
                            hyper1Descriptions(ih1x));
                        
            % Add to Results Class
            myResults.Add(iResults, ResultClass(descrip));
            fprintf("\nRunning model ...\n%s\n",descrip);
            
            % Get insResults
            insResults = resubPredict(mdl);
            
            % Evaluation Metrics of test set (In-sample)
            cv = crossval(mdl, 'KFold', 5);            
            insLoss = kfoldLoss(cv, 'LossFun', 'ClassifError'); 

            
            % Evaluation Metrics of test set (Out-of-sample)
            t2 = tic;
            [oosResults,cost] = predict(mdl, testData{:,incF});       
            tPredict = toc(t2);
            oosAcc = (sum(oosResults== testData{:,end}) / size(testData,1));
                        
            % Confusion Matrix
            C = confusionmat(testData{:,end}, oosResults);


            % Add the results to the collection
            myResults.m_results(iResults).m_trainTime = tTrain;
            myResults.m_results(iResults).m_predictTime = tPredict;
            %myResults.m_results(iResults).m_mdl = mdl;
            myResults.m_results(iResults).m_insLoss = round(insLoss,10);
            myResults.m_results(iResults).m_oosLoss = 1-round(oosAcc,10);
            myResults.m_results(iResults).m_cost = cost;
            myResults.m_results(iResults).m_insResults = insResults;
            myResults.m_results(iResults).m_oosResults = oosResults; 
            myResults.m_results(iResults).m_sampleidx = isx;           
            myResults.m_results(iResults).m_confusionMatrix = C;
            iResults = iResults + 1;
        end
    end 
end

% Random Forest Classification
for isx = 1:iSamples-1
    % Run each RF option for all features & reduced features
    featureDescriptions = {'All features','Reduced features'};

    for ifx = 1:2

        if ifx == 1
            incF = idxAllFeatures;                
        else
            incF = idxReducedFeatures;
        end

        hyper1Descriptions = {'Split criterian:gdi', 'Split criterian:deviance'};
        hyper1Parameters = {'gdi','deviance'};
        
        for ih1x = 1:2

            f = mySamples.m_samples(isx).m_trnFeatures(:,incF);
            c = mySamples.m_samples(isx).m_trnClasses;

            hyper2Descriptions = {'MinLeafSize:1','MinLeafSize:5','MinLeafSize:10'};
            hyper2Parameters = [1 5 10];
            for ih2x = 1:3

                    % Set up the description
                    descrip = strcat("Classifier:RF,", mySamples.m_samples(isx).m_description, ",", ...
                                    featureDescriptions(ifx),",", ...
                                    hyper1Descriptions(ih1x),",", ...
                                    hyper2Descriptions(ih2x));
                    
                    myResults.Add(iResults, ResultClass(descrip));                    
                    fprintf("\nRunning model ...\n%s\n",descrip);
                    
                    t1 = tic;
                    mdl = TreeBagger(25, f, c, ...
                        'NumPredictorsToSample', 5, ...
                        'SplitCriterion',char(hyper1Parameters(ih1x)), ...
                        'MinLeafSize', hyper2Parameters(ih2x), ...
                        'Method','classification', ...
                        'OOBPrediction', 'on');
                    tTrain = toc(t1);
                    % Evaluation Metrics of test set (In-sample)
                    oobErrorBaggedEnsemble = oobError(mdl);
                    oobResults = oobPredict(mdl);
                    
                    % Evaluation Metrics of test set (Out-of-sample)
                    t2 = tic;
                    [result,cost] = predict(mdl, testData{:,incF});
                    tPredict = toc(t2);
                    
                    % transform result into same form as testdata
                    oosResults2 = zeros(size(testData{:,end}));
                    oosResults2 = str2double(result);                    
                    oosAcc = (sum(oosResults2 == testData{:,end}) / size(testData,1));

                    % Confusion Matrix
                    C = confusionmat(testData{:,end}, oosResults2);

                    % Add the results to the collection
                    myResults.m_results(iResults).m_trainTime = tTrain;
                    myResults.m_results(iResults).m_predictTime = tPredict;
                    myResults.m_results(iResults).m_insLoss = oobErrorBaggedEnsemble;
                    myResults.m_results(iResults).m_oosLoss = 1-round(oosAcc,10);
                    myResults.m_results(iResults).m_cost = cost;
                    %myResults.m_results(iResults).m_mdl = mdl;
                    myResults.m_results(iResults).m_insResults = oobResults ;
                    myResults.m_results(iResults).m_oosResults = oosResults2; 
                    myResults.m_results(iResults).m_sampleidx = isx;           
                    myResults.m_results(iResults).m_confusionMatrix = C;
                    iResults = iResults + 1;
            end
        end
    end 
end
