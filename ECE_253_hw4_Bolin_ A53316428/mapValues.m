function [im_seg] = mapValues(im,idx)
    N1 = reshape(im(:,:,1),[],1);
    N2 = reshape(im(:,:,2),[],1);
    N3 = reshape(im(:,:,3),[],1);
    features = [N1,N2,N3]; 
    [a,b,c] = size(im);
    im_seg=zeros(a*b,c);
    
   for i = 1:7
       idx2 = find(idx == i);
       Mean = mean(features(idx2,:));
       for j = 1:length(idx2)
        im_seg(idx2(j),:) = Mean;  
       end
   end   
   
%    Plot
   im_seg = reshape(im_seg,[720,1280,3]);
   imshow(uint8(im));
   title('Original Image');
   figure;
   imshow(uint8(im_seg));
   title('Image after segmentation');
end
