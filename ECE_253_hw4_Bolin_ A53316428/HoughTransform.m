function [rho,H] = HoughTransform(img)
    % H: HT transform matrix
    
    % Initialization
    [x,y] = size(img);
    theta = [-90:90];
    len = ceil(sqrt((x-1)^2+(y-1)^2));
    [M,N] = find(img);
    rho = zeros(length(x),2*90+1);
    H = zeros(2*len+1,2*90+1);
    
    % Calcualte rho
    for i = 1:length(M)
        rho(i,:) = round((M(i)-1).*cosd(theta)+(N(i)-1).*sind(theta));
    end
    
    % Calculate H
    for i = 1:length(M)
        for j = 1:181
            x_idx = rho(i,j)+len;
            H(x_idx,j) = H(x_idx,j)+1;
        end
    end
    
    Hs = H./max(H(:));
    imshow(Hs,[],'XData',theta,'YData',-len:len,'InitialMagnification','fit');
    xlabel('\theta'), ylabel('\rho');
    colorbar;colormap(gray);
    axis on, axis normal

end






