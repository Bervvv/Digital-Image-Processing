function [MSE_Uni] = myF(I,s)
    I = double(I);
    [x,y]= size(I);
    Store = zeros(x,y);
    interval = 255./2.^s;
    for i = 1:length(s)
        T = round(0:interval(i):255); 
        for j = 1:length(T)-1
            Store(I>T(j) & I<=T(j+1)) = (T(j)+T(j+1))/2;
        end
        subplot(2,4,i+1)
        Store = uint8(Store);
        imshow(Store)
        title(['bit=',num2str(i)])
        MSE_Uni(i) = sum(sum((double(Store)-I).^2))/numel(I);
    end
end


