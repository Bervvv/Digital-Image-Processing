function [ ] = CED(img,te)
% Smoothing 
I = double(img);
k = 1/159*[2 4 5 4 2;4 9 12 9 4;5 12 15 12 5;4 9 12 9 4; 2 4 5 4 2];
I_filter = imfilter(I,k);

% Finding Gradients
kx = [-1 0 1;-2 0 2;-1 0 1];
ky = [-1 -2 -1;0 0 0;1 2 1];

Gx = imfilter(I_filter,kx);
Gy = imfilter(I_filter,ky);
G = sqrt(Gx.^2+Gy.^2);
G_theta = atan(Gy./Gx);

% Non-maximum Suppression (NMS)
[row,col] = size(I_filter);

for i = 1:row
    for j =1:col
        if G_theta(i,j) <= pi/8 && G_theta(i,j) >=-pi/8
            G_theta(i,j) = 0;
        elseif G_theta(i,j) <= 3/8*pi && G_theta(i,j) > pi/8
            G_theta(i,j) = pi/4;
        elseif G_theta(i,j) < -pi/8 && G_theta(i,j) >= -3/8*pi
            G_theta(i,j) = -pi/4;
        elseif G_theta(i,j) < -3/8*pi && G_theta(i,j) >= -1/2*pi
            G_theta(i,j) = pi/2;
        elseif G_theta(i,j) <= 1/2*pi && G_theta(i,j) > -3/8*pi
            G_theta(i,j) = pi/2;
        end
    end
end

I_pad = padarray(G,[1,1],0,'both');
G_theta = padarray(G_theta,[1,1],0,'both');
[row_pad,col_pad] = size(I_pad);
I_nms = zeros(row_pad,col_pad);

for i = 2:row_pad-1
    for j = 2:col_pad-1
        if G_theta(i,j) == 0 
            if I_pad(i,j+1) > I_pad(i,j) || I_pad(i,j-1) > I_pad(i,j)
                I_nms(i,j) = 0;
            else
                I_nms(i,j) = I_pad(i,j);
            end
            elseif G_theta(i,j) == 1/4*pi
                if I_pad(i+1,j-1) > I_pad(i,j) || I_pad(i-1,j+1) > I_pad(i,j)
                    I_pad(i,j) = 0;
                else
                    I_nms(i,j) = I_pad(i,j);
                end
            elseif G_theta(i,j) == -1/4*pi
                if I_pad(i-1,j-1) > I_pad(i,j) || I_pad(i+1,j+1) > I_pad(i,j)
                    I_pad(i,j) = 0; 
                else
                    I_nms(i,j) = I_pad(i,j);
                end
            elseif G_theta(i,j) == pi/2
                if I_pad(i-1,j) > I_pad(i,j) || I_pad(i+1,j) > I_pad(i,j)
                    I_pad(i,j) = 0;
                else
                    I_nms(i,j) = I_pad(i,j);
                end
        end
    end
end

            
% Thresholding
I_edge = zeros(row_pad,col_pad);
for i = 1:row_pad
    for j =1:col_pad
        if I_nms(i,j) < te
           I_edge(i,j) = 0;
        else
           I_edge(i,j) = I_nms(i,j);
        end
    end
end


G = imshow(uint8(G));
title('The original gradient magnitude image')
figure
I_nms = imshow(uint8(I_nms));
title('The image after NMS')
figure
I_edge = imshow(uint8(I_edge));
title('The image after thresholding')
end

