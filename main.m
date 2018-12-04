clear all;
close all;
clc;

%% Simple Image Compression With PCA (Patch Version)

% Read image file
I = im2single(rgb2gray(imread('./lenna.png')));
[nH, nW] = size(I); 

% Create patches
pSize = [16,16];
P = zeros(nH/pSize(1)*nW/pSize(2), pSize(1)*pSize(2)); % P:pathces (N,dim)
n = 1;
for h = 1:pSize(1):nH
    for w = 1:pSize(2):nW
        patch = I(h:h+pSize(1)-1, w:w+pSize(2)-1);
        P(n,:) = patch(:); % 2D patch data => 1D data as X dim
        n = n + 1;
    end
end
U = repmat(mean(P,1),[size(P,1),1]);
X = P - U;
% Show patches (Heavy Process)
% figure;
% for n=1:1:size(X,1)
%     subplot(nH/patch_size(1), nW/patch_size(2), n);
%     imshow(reshape(P(n,:),[16,16]));
% end

% PCA for X
covX = X'*X / (size(X,2)-1);
[V,D] = eig(covX); % covB*V = V*D, V:principal components (row vector), D:coefficients
[~,ind] = sort(diag(D),'descend');
V = V(:,ind); % sort principal components in descending order of coefficients

% Show principal components from Image Patches
figure('position', [1,200,1440,267]);
colormap gray
for i = 1:1:20
    subplot(2,10,i);
    imagesc(reshape(V(:,i),pSize)); 
    axis off; title(['PC:',num2str(i)])
end

% Reconstruct image
ln = [1, 5, 10, 25, 50, size(X,2)]; % latent number for reducing data dimension
rP = cell(numel(ln),1);
rI = cell(numel(ln),1);
for i = 1:1:numel(ln) 
    Z  = X*V(:,1:ln(i));  % Z:pricipal projections (score) (data dimension is reduced by ln(i))
    rX = Z*V(:,1:ln(i))'; % reconstruction from Z, X=X*V*V'=Z*V' (V*V'=I, V'=V^(-1), V is orthogonal matrix (íºçsçsóÒ))
    rP{i} = rX + U;
    rI{i} = zeros(size(I));    
    n = 1;
    for h = 1:pSize(1):nH
        for w = 1:pSize(2):nW
            rpatch = reshape(squeeze(rP{i}(n,:,:)),pSize(1),pSize(2));
            rI{i}(h:h+pSize(1)-1, w:w+pSize(2)-1) = rpatch;
            n = n + 1;
        end
    end

end

% Show reconstruct image
figure;
subplot(3,3,1);
imshow(I); title(['input image *(h,w)=(',num2str(size(I,1)),',',num2str(size(I,2)),')']);
for i = 1:1:numel(ln)
    subplot(3,3,i+3);
    imshow(rI{i}); title(['Using PC:1-',num2str(ln(i))]);
end
