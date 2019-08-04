% Author: JM
% Date: 19 Nov 2018
% Purpose: Class to hold a Sample, features and target class
classdef SampleClass < handle

    properties
        m_trnFeatures
        m_trnClasses  
        m_description
        m_items
        m_features
    end
    
    
    methods        
        function SC =  SampleClass(f,c, description)            
            SC.m_trnFeatures = f;
            SC.m_trnClasses = c;
            SC.m_description = description;
            
            [SC.m_items,SC.m_features] = size(f);
        end        
    end
end
