% Author: JM
% Date: 20 Nov 2018
% Purpose: collection of resultclass
classdef ResultsClass < handle

    properties
        m_results    
        m_numberResults
    end
        
    methods        
        function RCS = ResultsClass(numberResults)            
            RCS.m_numberResults = numberResults;
            RCS.m_results = ResultClass.empty(RCS.m_numberResults,0);
        end 
        
        function RCS= Add(RCS, idx, resultClass)
            RCS.m_results(idx) = resultClass;
        end
    end
end