% Author: JM
% Date: 20 Nov 2018
% Purpose: Class to hold results of a classifier
classdef ResultClass < handle

    properties
        m_description
        m_insLoss
        m_oosLoss
        m_cost
        m_confusionMatrix        
        m_trainTime
        m_predictTime
        m_insResults
        m_oosResults
        m_sampleidx
    end
    
    methods
        
        function RC = ResultClass(description)
            RC.m_description = description;
        end
        
        function insAcc = CalculateInSampleAccuracy(RC, trnLabels)
            if isa(trnLabels,'table')
                trnLabels = table2array(trnLabels);
            end
            
            results2 = zeros(numel(trnLabels),1);
            results2 = str2double(RC.m_insResults);
            insAcc = sum(results2 == trnLabels)/numel(trnLabels);
        end
        
        function cost = CalculateFinancialCost(RC)
            TP = RC.m_confusionMatrix(2,2);
            
            FP = RC.m_confusionMatrix(1,2);
            FN = RC.m_confusionMatrix(2,1);
            
            cost = (122.21*FN) + (32.84*FP) + (32.84*TP);
        end
        
        function f1 = CalculateF1Score(RC)
            TP = RC.m_confusionMatrix(2,2);
            
            FP = RC.m_confusionMatrix(1,2);
            FN = RC.m_confusionMatrix(2,1);
            
            f1 = 2*TP /((2*TP)+FP+FN);
        end
        
    end
end
