%% Bolin He, A53316428, Nov 19 2019
%% Q1
close all;
clear all;
clc;

img = imread('geisel.jpg');
img = rgb2gray(img);
CED(img,70)
fprintf('I choose the thresholding value as 70.')

% function [ ] = CED(img,te)
% % Smoothing 
% I = double(img);
% k = 1/159*[2 4 5 4 2;4 9 12 9 4;5 12 15 12 5;4 9 12 9 4; 2 4 5 4 2];
% I_filter = imfilter(I,k);
% 
% % Finding Gradients
% kx = [-1 0 1;-2 0 2;-1 0 1];
% ky = [-1 -2 -1;0 0 0;1 2 1];
% 
% Gx = imfilter(I_filter,kx);
% Gy = imfilter(I_filter,ky);
% G = sqrt(Gx.^2+Gy.^2);
% G_theta = atan(Gy./Gx);
% 
% % Non-maximum Suppression (NMS)
% [row,col] = size(I_filter);
% 
% for i = 1:row
%     for j =1:col
%         if G_theta(i,j) <= pi/8 && G_theta(i,j) >=-pi/8
%             G_theta(i,j) = 0;
%         elseif G_theta(i,j) <= 3/8*pi && G_theta(i,j) > pi/8
%             G_theta(i,j) = pi/4;
%         elseif G_theta(i,j) < -pi/8 && G_theta(i,j) >= -3/8*pi
%             G_theta(i,j) = -pi/4;
%         elseif G_theta(i,j) < -3/8*pi && G_theta(i,j) >= -1/2*pi
%             G_theta(i,j) = pi/2;
%         elseif G_theta(i,j) <= 1/2*pi && G_theta(i,j) > -3/8*pi
%             G_theta(i,j) = pi/2;
%         end
%     end
% end
% 
% I_pad = padarray(G,[1,1],0,'both');
% G_theta = padarray(G_theta,[1,1],0,'both');
% [row_pad,col_pad] = size(I_pad);
% I_nms = zeros(row_pad,col_pad);
% 
% for i = 2:row_pad-1
%     for j = 2:col_pad-1
%         if G_theta(i,j) == 0 
%             if I_pad(i,j+1) > I_pad(i,j) || I_pad(i,j-1) > I_pad(i,j)
%                 I_nms(i,j) = 0;
%             else
%                 I_nms(i,j) = I_pad(i,j);
%             end
%             elseif G_theta(i,j) == 1/4*pi
%                 if I_pad(i+1,j-1) > I_pad(i,j) || I_pad(i-1,j+1) > I_pad(i,j)
%                     I_pad(i,j) = 0;
%                 else
%                     I_nms(i,j) = I_pad(i,j);
%                 end
%             elseif G_theta(i,j) == -1/4*pi
%                 if I_pad(i-1,j-1) > I_pad(i,j) || I_pad(i+1,j+1) > I_pad(i,j)
%                     I_pad(i,j) = 0; 
%                 else
%                     I_nms(i,j) = I_pad(i,j);
%                 end
%             elseif G_theta(i,j) == pi/2
%                 if I_pad(i-1,j) > I_pad(i,j) || I_pad(i+1,j) > I_pad(i,j)
%                     I_pad(i,j) = 0;
%                 else
%                     I_nms(i,j) = I_pad(i,j);
%                 end
%         end
%     end
% end
% 
%             
% % Thresholding
% I_edge = zeros(row_pad,col_pad);
% for i = 1:row_pad
%     for j =1:col_pad
%         if I_nms(i,j) < te
%            I_edge(i,j) = 0;
%         else
%            I_edge(i,j) = I_nms(i,j);
%         end
%     end
% end
% 
% 
% G = imshow(uint8(G));
% title('The original gradient magnitude image')
% figure
% I_nms = imshow(uint8(I_nms));
% title('The image after NMS')
% figure
% I_edge = imshow(uint8(I_edge));
% title('The image after thresholding')
% end



%% Q2
close all;
clear all;
clc;

%%%%% Image Car %%%%%
car = imread('Car.tif');
imagesc(car);
colormap(gray);
colorbar;
title('Original Image');

car = double(car);
car_pad = padarray(car,[133,172],0,'both');
imFFT = fft2(car_pad);
imFFT = fftshift(imFFT);
figure
imagesc(-256:255,-256:255,log(abs(imFFT)));
colorbar;
title('Image after FFT')
xlabel('u');
ylabel('v');

% Notch Filter
D0 = 30;
n = 4;
Position =[-90,-167;-90,-82;-82,87;-82,173];
uk = Position(:,1);
vk = Position(:,2);
H_NR = 1;
[u,v] = meshgrid(-256:255);
for k = 1:4
    Dk1=sqrt(((u-uk(k)).^2)+((v-vk(k)).^2));
    Dk2=sqrt(((u+uk(k)).^2)+((v+vk(k)).^2));
    H_NR=H_NR.*(1./(1+(D0./Dk1).^(2*n))).*(1./(1+(D0./Dk2).^(2*n)));
end
figure
imagesc(-256:255,-256:255,abs(H_NR)); 
colorbar;
colormap(gray);
title('Butterworth Notch Reject Filter');
xlabel('u'); 
ylabel('v');

% Apply Filter
figure
imagesc(-256:255,-256:255,log(abs(imFFT.*H_NR)));
title('Image after FFT with Notch Filter');
colorbar;
xlabel('u'); 
ylabel('v');

imfilter = uint8(ifft2(ifftshift(imFFT.*H_NR)));
figure
imagesc(imfilter(134:379,173:340))
colormap(gray);
title('Filtered Image')
colorbar;

%%%%% Image Street %%%%%
street = imread('Street.png');
figure;
imagesc(street);
colorbar;
colormap(gray);
title('Original Image');

street = double(street);
street_pad = padarray(street,[180,153],0,'post');
imFFT2 = fftshift(fft2(street_pad));
figure;
imagesc(-256:255,-256:255,log(abs(imFFT2)));
colorbar;
xlabel('u');
ylabel('v');

% Notch Filter
D0 = 20;
n = 3;
Position2 =[0,164;-168,0];
uk = Position2(:,1);
vk = Position2(:,2);
H_NR2 = 1;
[u,v] = meshgrid(-256:255);
for k = 1:2
    Dk1=sqrt(((u-uk(k)).^2)+((v-vk(k)).^2));
    Dk2=sqrt(((u+uk(k)).^2)+((v+vk(k)).^2));
    H_NR2=H_NR2.*(1./(1+(D0./Dk1).^(2*n))).*(1./(1+(D0./Dk2).^(2*n)));
end
figure
imagesc(-256:255,-256:255,abs(H_NR2)); 
colorbar;
colormap(gray);
title('Butterworth Notch Reject Filter');
xlabel('u'); 
ylabel('v');

% Apply Filter
figure
imagesc(-256:255,-256:255,log(abs(imFFT2.*H_NR2)));
title('Image after FFT with Notch Filter');
colorbar;
xlabel('u'); 
ylabel('v');

imfilter2 = uint8(ifft2(ifftshift(imFFT2.*H_NR2)));
figure
imagesc(imfilter2(1:332,1:359))
colormap(gray);
title('Filtered Image')
colorbar;

%
Para1 = {'n';'D0';'u1';'v1';'u2';'v2';'u3';'v3';'u4';'v4'};
Para2 = {'n';'D0';'u1';'v1';'u2';'v2'};
num1 = [4;30;-90;-167;-90;-82;-82;87;-82;173];
num2 = [3;20;0;164;-168;0];   
Parameter_in_2i = table(categorical(Para1),num1)
Parameter_in_2ii = table(categorical(Para2),num2)

