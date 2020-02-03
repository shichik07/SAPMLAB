function [binary_array] = convert_sample(rej_art,sampleinfo)
% Simple function with input of the visually identified samples to be
% rejected and a second input containing the sampples that are contained
% each trial. The function simply returns a logical array that contains all
% trial indices that were slected to be rejected, based on the rejected
% sample snippets.
    trls                    = size(rej_art,1);
    smpls                    = size(sampleinfo,1);
    binary_array            = [];
    a = [];
    for i = 1:trls
        cur = [rej_art(i,1):rej_art(i,2)];
        a = cat(2,a,cur);
    end

    for ii = 1:smpls
        period = [sampleinfo(ii,1):sampleinfo(ii,2)];
        if any(ismember(a, period))
            binary_array = [binary_array, ii];
        end
    end
    ind = unique(binary_array); %get unique indices
    disp([num2str(length(ind)), ' trials have been marked for rejection.'])
    binary_array = zeros(size(sampleinfo,1),1);
    binary_array(ind)= 1;
    binary_array = boolean(binary_array);
        
                