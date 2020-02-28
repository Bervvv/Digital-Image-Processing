function [h] = compute_norm_rgb_histogram(input)
% input (RGB/color image) and one output (1 x 96 vector)
I = imread(input);
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

% Initialize
[x,y] = size(R);
n = 255/32;
h = zeros(1,3*32);

%  Count the numbers
for bins = 1:32
    for i = 1:x
        for j = 1:y
            if R(i,j)>n*(bins-1) && R(i,j)<=n*bins
                h(bins) = h(bins)+1;
            end
        end
    end
end

for bins = 33:64
    for i = 1:x
        for j = 1:y
            if G(i,j)>n*(bins-33) && G(i,j)<=n*(bins-32)
                h(bins) = h(bins)+1;
            end
        end
    end
end


for bins = 65:96
    for i = 1:x
        for j = 1:y
            if B(i,j)>n*(bins-65) && B(i,j)<=n*(bins-64)
                h(bins) = h(bins)+1;
            end
        end
    end
end

% Plot the histogram
h = h./sum(h);
bk = zeros(1,32); % blank matrix
bar([h(1,1:32),bk,bk],'r')
hold on
bar([bk,h(1,33:64),bk],'g')
hold on
bar([bk,bk,h(1,65:96)],'b')
legend('Red','Green','Blue')
xlabel('Channel')
ylabel('Probability (%)')



