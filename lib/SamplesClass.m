% Author: JM
% Date: 20 Nov 2018
% Purpose: collection of samples
classdef SamplesClass < handle

    properties
        m_samples     
        m_numberSamples
    end
    
    
    methods        
        function SCS = SamplesClass(numberSamples)            
            SCS.m_numberSamples = numberSamples;
            SCS.m_samples = SampleClass.empty(SCS.m_numberSamples,0);
        end 
        
        function SCS = Add(SCS, idx, sampleClass)
            SCS.m_samples(idx) = sampleClass;
        end
        
    end
end