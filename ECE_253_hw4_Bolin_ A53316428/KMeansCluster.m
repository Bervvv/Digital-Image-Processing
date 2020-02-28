function [idx, centers] = kMeansCluster(features, centers)

    [M,N] = size(features);
    [center_row,center_col] = size(centers);
    
% Iterate for 100 times
    % Find the nearest center
    for i = 1:100;
        Distance = pdist2(features,centers);
        [~,idx] = min(Distance,[],2);
        centers = zeros(center_row,3);
        % Recalculate the center
        for j = 1:center_row
            center_avg = find(idx == j);  
            centers(j,:) = mean(features(center_avg,:));              
        end
    end
end








