function [ ] = AHE(im,win_size)
impad = padarray(im,[(win_size-1)/2,(win_size-1)/2],'symmetric');
[x,y] = size(im);
output = zeros(x,y);

for i = 1:x
    for j =1:y
        rank = 0;
%         contextual_region = im(i:i+win_size-1,j:j+win_size-1);
        for k = i:i+win_size-1
            for t = j:j+win_size-1
                 if impad((win_size-1)/2+i,(win_size-1)/2+j) > impad(k,t)
                     rank = rank+1;
                 end
            end
        end  
        output(i,j) = rank*255/(win_size*win_size);
    end
end
 
imshow(output,[])
