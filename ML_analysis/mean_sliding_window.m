function [data_array] = mean_sliding_window(window_size, overlap, dats, time)
%window_size in ms
%overlap in ms
%data array to be downsampled
%time sampling information for decomposition in Hz

if nargin ~= 4
    msg = ['Not enough inputs to perform a slinding' ...
        ' window decomposition'];
    error(msg)
end

windowsize      = windowsize/(1000/time);
overlap         = overlap/(1000/time);
steps           = round(size(dats,3)/overlap) - 1; % otherwise we have overlap to a nonexisting sample
data_array      = zeros(size(dats,1), size(dats,2), steps);
former_step     = 1; 


for i = 1:steps
    data_array(:, :, i) = mean(dats(:, :, former_step:[former_step + windowsize]), 3);
    former_step         = former_step + overlap; 
end


    
