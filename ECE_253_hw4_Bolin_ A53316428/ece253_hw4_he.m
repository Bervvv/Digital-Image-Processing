%% Bolin He, PID: A53316428, Hw04
% Dec 3,2019

%% Q1 Hough Transform
clear all;
clc;
% i
% function [rho,H] = HoughTransform(img)
%     % H: HT transform matrix
%     
%     % Initialization
%     [x,y] = size(img);
%     theta = [-90:90];
%     len = ceil(sqrt((x-1)^2+(y-1)^2));
%     [M,N] = find(img);
%     rho = zeros(length(x),2*90+1);
%     H = zeros(2*len+1,2*90+1);
%     
%     % Calcualte rho
%     for i = 1:length(M)
%         rho(i,:) = round((M(i)-1).*cosd(theta)+(N(i)-1).*sind(theta));
%     end
%     
%     % Calculate H
%     for i = 1:length(M)
%         for j = 1:181
%             x_idx = rho(i,j)+len;
%             H(x_idx,j) = H(x_idx,j)+1;
%         end
%     end
%     
%     Hs = H./max(H(:));
%     imshow(Hs,[],'XData',theta,'YData',-len:len,'InitialMagnification','fit');
%     xlabel('\theta'), ylabel('\rho');
%     colorbar;colormap(gray);
%     axis on, axis normal
% 
% end


% ii
test_img = [1 0 0 0 0 0 0 0 0 0 1
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0 0 0 1];

imshow(test_img,'InitialMagnification',5000);
title('Original image');
colorbar;
figure;
HoughTransform(test_img);
title('HT image');
colorbar;

% select point (45,0) and (-45,7)
figure;
hold on;
xx = 0:10;
y1 = round(0/sind(45)-cosd(45)/sind(45).*xx)+10;
y2 = round(7/sind(-45)-cosd(-45)/sind(-45).*xx)+10;
for i = 1:length(xx)
        test_img(xx(i)+1,y1(i)+1) = 1;
        test_img(xx(i)+1,y2(i)+1) = 1;
end
imshow(test_img,'InitialMagnification',5000)
colorbar;
title('Original image with lines')

% iii
lane = imread('lane.png');
lane2 = double(rgb2gray(lane));
E = edge(lane2,'sobel');
title('Image after Sobel Filter');
figure
imshow(lane)
title("Original image");
colorbar;
figure
imshow(E)
title("Binary edge image");
colorbar;
figure
[rho2, H2] = HoughTransform(E);
title("HT")
colorbar;

% Threshold
[A,B] = find(H2 > 0.75*max(H2(:)))
[row,col] = size(E);
len = ceil(sqrt((row-1)^2+(col-1)^2));
Amap = A-len;
Bmap = B-91;
L = zeros(length(A),2);
figure
imshow(lane);
hold on;

for i=1:length(A)
    L(i,:)=[round((Amap(i))*sind(Bmap(i))+1),round((Amap(i))*cosd(Bmap(i))+1)]; 
    j=1:col;
    k=-(1/cotd(Bmap(i)))*(j-L(i,1))+L(i,2);
plot(j,k,'r','linewidth',1)
set(gca,'ydir','reverse');
ylim([1 row]);xlim([1 col])
end

title('Original image with lines');
colorbar;

% iv
figure()
imshow(lane);
colorbar;
hold on;
% Select theta as [-30,40]U[30,40]
drive = find(abs(Bmap)>=30 & abs(Bmap)<=40);
L2 = zeros(length(A),2);
for i=1:length(drive)
    L2(i,:)=[round((Amap(drive(i)))*sind(Bmap(drive(i)))+1),round((Amap(drive(i)))*cosd(Bmap(drive(i)))+1)]; 
    j=1:col;
    k=-(1/cotd(Bmap(drive(i))))*(j-L2(i,1))+L2(i,2);
   plot(j,k,'g','linewidth',2)
   set(gca,'ydir','reverse');
   ylim([1 row]);xlim([1 col])
end   
title('Driver lane only')
fprintf('We choose theta as [-30,40]U[30,40]');

%% Q2 K-Means Segmentation 
close all;
clear all;
clc;

im = imread('white-tower.png');
im = double(im);

features = createDataset(im);

nclusters = 7;
id = randi(size(features,1),1,nclusters);
centers = features(id,:);

[idx, centers] = KMeansCluster(features,centers);

im_seg = mapValues(im, idx);

% Print centers
fprintf('The centers of seven clusters are:\n')
table(centers)


% function features = createDataset(im)
%     N1 = reshape(im(:,:,1),[],1);
%     N2 = reshape(im(:,:,2),[],1);
%     N3 = reshape(im(:,:,3),[],1);
%     features = [N1,N2,N3]; 
% end
 
 
% function [idx, centers] = kMeansCluster(features, centers)
% 
%     [M,N] = size(features);
%     [center_row,center_col] = size(centers);
%     
% % Iterate for 100 times
%     % Find the nearest center
%     for i = 1:100;
%         Distance = pdist2(features,centers);
%         [~,idx] = min(Distance,[],2);
%         centers = zeros(center_row,3);
%         % Recalculate the center
%         for j = 1:center_row
%             center_avg = find(idx == j);  
%             centers(j,:) = mean(features(center_avg,:));              
%         end
%     end
% end


% function [im_seg] = mapValues(im,idx)
%     N1 = reshape(im(:,:,1),[],1);
%     N2 = reshape(im(:,:,2),[],1);
%     N3 = reshape(im(:,:,3),[],1);
%     features = [N1,N2,N3]; 
%     [a,b,c] = size(im);
%     im_seg=zeros(a*b,c);
%     
%    for i = 1:7
%        idx2 = find(idx == i);
%        Mean = mean(features(idx2,:));
%        for j = 1:length(idx2)
%         im_seg(idx2(j),:) = Mean;  
%        end
%    end   
%    
% %    Plot
%    im_seg = reshape(im_seg,[720,1280,3]);
%    imshow(uint8(im));
%    title('Original Image');
%    figure;
%    imshow(uint8(im_seg));
%    title('Image after segmentation');
% end

















