function [p,omega,obj] = computeBarycenterDerivativeMultiGPU(Dictionary,q,xi_original,lambdas,niter,gamma)
% Compute derivative of loss between input q and Wasserstein barycenter of
% Dictionary with weights lambda.
% - Dictionary: tensor of dimensions [ i_1 ... i_n] where i_n is the number of
% points in the Dictionary. Each slice (:,:,...,j) of that tensor is in the
% simplex (sums to 1).
% - q: input (tensor of size [i_1 ... i_n-1 ] that sums to 1)
% - xi_original: kernel operator
% - lambdas: lambda vector of weights. Should be a vector in the simplex of
% size i_n
% - niter: number of iterations in Sinkhorn loop.
% - gamma: entropic smoothing.

%   Copyright (c) 2016 Marco Cuturi

mylog = @(x)log( max(1e-300, x) );

DIMENSIONS=size(Dictionary);

if Wlossweight>0, % keep one GPU to run Sinkhorn...
    gpusUsed=gpuDeviceCount-1;
else
    gpusUsed=gpuDeviceCount;
end


gpusUsed=min(gpusUsed,DIMENSIONS(end)); % no need to have more workers than data points...

%disp(['Using ',num2str(gpusUsed),' GPUs --- ']);

% SPLIT

I=evenSplit(1:DIMENSIONS(end),gpusUsed);
% parpools. always destroy.
if isempty(gcp('nocreate')) % no pool
    %poolobj = gcp('nocreate');
    parpool(gpusUsed); % one worker for each GPU.
end


fprintf('-*');
spmd
    %allGpus=
    %disp(['Ind: ',num2str(labindex),' 1']);
    gpuDevice(labindex);
    b = ones([DIMENSIONS(1:end-1) length(I{labindex})],'gpuArray');
    matrixDIMENSIONS=[prod(DIMENSIONS(1:end-1)),length(I{labindex})];
    %a = b;
    Q=gpuArray(Dictionary(:,:,:,I{labindex}));
    lambda=gpuArray(lambdas(I{labindex}));
    b1=cell(niter+1,1);
    % phis=cell(niter); don't store phis, not enough memory
    b1{1}=b;
    Hx=gpuArray(xi_original.Hx);
    Hy=gpuArray(xi_original.Hy);
    Hz=gpuArray(xi_original.Hz);
    xi=xi_original.xi;
    
    for i=1:niter
        
        %     if verb
        %         progressbar(i,niter);
        %         %displayMemory; drawnow;
        %     end
        %     spmd
        
        % update right marginals
        
        %Err_q(i) = Err_q(i) + sum(squeeze(sum(sum( (a .* xi(b) - Q).^2,1,2)))./normsQ;
        %[size(Q), size(b)]
        %disp(['Ind: ',num2str(labindex),' 1']);
        phi = xi(Q./ xi(b,Hx,Hy,Hz),Hx,Hy,Hz);
        %disp(['Ind: ',num2str(labindex),'  ',num2str(phi(1))]);
        % phis{i}=phi; don't store them.
        
        % update barycenter by geometric mean
        %logp = gather(reshape(reshape(log(phi),matrixDIMENSIONS)*lambda,DIMENSIONS(1:end-1)));
        logp = vec(reshape(mylog(phi),matrixDIMENSIONS)*lambda);
        
        labBarrier;
        
        if labindex<gpusUsed,
            labSend(gather(logp),gpusUsed);
            p=labReceive(gpusUsed);
        else
            for jj=1:gpusUsed-1,
                logp=logp+labReceive;%(jj); same, should work faster.
            end
            
            p=exp(logp); p=p/sum(p(:));
            p=gather(reshape(p,size(q)));
            if isnan(p(1)) || imag(p(1))>0,
                error(['regularization too weak, computation of W bar derivative blew up after ',num2str(i),' iterations']);
            end
            for jj=1:gpusUsed-1, % compute common barycenter
                labSend(p,jj)
            end
        end
        p=gpuArray(p);
        % update left marginals
        b = reshape(bsxfun(@times, 1./ reshape(phi, matrixDIMENSIONS),p(:)),size(b));
        b1{i+1} = b; % previous b value
        %pp = [];
        %Mem=allGpus.AvailableMemory/allGpus.TotalMemory
    end
end
p=gather(p{1});

gpuDevice(gpuDeviceCount); % better not have anything on that one....
% compute Sinkhorn divergence between current estimate of barycenter p and input q)
[objW,a]=simpleSinkhorn_(p,q,xi_original); 
objW=objW*gamma;

% gradient of the regularized W w.r.t. p is equal to log(a), the scaling
grad=gamma*log(a); 
obj=gather(objW);

% Unfold the sinkhorn loop using the scalings stored previously to 
% compute now the gradient
grad = grad.*p;

% computing
spmd
    for i=niter:-1:1
        %size(Q)
        %size(b1{i})
        
        xib=xi(b1{i},Hx,Hy,Hz);
        %disp(['Ind: ',num2str(labindex),' 2',' Iter ',num2str(i)]);
        
        phis=xi(Q./xib,Hx,Hy,Hz); % can't store'em, not enough memory.
        if i==niter,
            omega= (grad(:)'*reshape(mylog(phis),matrixDIMENSIONS))';
        else
            omega= omega+(grad(:)'*reshape(log(phis),matrixDIMENSIONS))';
        end
        %disp(['Ind: ',num2str(labindex),' 2']);
        if i==niter,
            r=-b1{i}.* xi( ...
                xi( reshape(vec(grad)*lambda',size(xib))./phis,Hx,Hy,Hz)...
                .* (Q./(xib.^2)) ...
                ,Hx,Hy,Hz);
        else
            r=-b1{i}.* xi( ...
                xi( (reshape(bsxfun(@minus,vec(grad)*lambda',vec(r)),size(xib)))./phis,Hx,Hy,Hz)...
                .* (Q./(xib.^2)) ...
                ,Hx,Hy,Hz);
        end
        %b1{i}=[]; % free some memory
        %disp(['Ind: ',num2str(labindex),' 3']);
        partialsumsr=squeeze(sum(r,length(DIMENSIONS)));
        %Mem=allGpus.AvailableMemory/allGpus.TotalMemory
                
        labBarrier; % wait until all labs have reached execution.
        
        % share now computations executed on the gpusUsed-1 gpus, 
        % centralize on the last GPU.
        if labindex<gpusUsed, 
            labSend(gather(partialsumsr),gpusUsed);
            r=labReceive(gpusUsed);
        else            
            for jj=1:gpusUsed-1,
                partialsumsr=partialsumsr+labReceive;%(jj); % should work better, any order is fine
            end
            
            r=gather(reshape(partialsumsr,size(q)));
            if isnan(r(1)) || imag(r(1))>0,
                error(['regularization too weak, computation of W bar derivative blew up after ',num2str(i),' iterations']);
            end
            for jj=1:gpusUsed-1, % compute common barycenter
                labSend(r,jj)
            end
        end
        r=gpuArray(r);
    end
end
omegas=omega{1};
for j=2:gpusUsed, % forgot why I am not doing a cell2mat :)
    omegas=[omegas;omega{j}];
end
omega=gather(omegas);


