function [out] = Transform(in, typeTrans)

    switch typeTrans
        case 'posLog'
            out = log10(in + abs(min(in)) + 1);
        case 'negLog'
            out = log10(max(in)+1-in);             
        % add other transformation here
    end
    
end