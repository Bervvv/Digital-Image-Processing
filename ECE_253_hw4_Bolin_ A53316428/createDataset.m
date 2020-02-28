function features = createDataset(im)
    N1 = reshape(im(:,:,1),[],1);
    N2 = reshape(im(:,:,2),[],1);
    N3 = reshape(im(:,:,3),[],1);
    features = [N1,N2,N3]; 
end

