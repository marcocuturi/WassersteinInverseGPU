function [val,left,right] = simpleSinkhorn_(p,q,xi)

Hx=gpuArray(xi.Hx);
Hy=gpuArray(xi.Hy);
Hz=gpuArray(xi.Hz);


right=ones(size(p));
criterion=inf;
while criterion>1e-4;
    for pipo=1:100,
        left=p./xi.xi(right,Hx,Hy,Hz);
        right=q./xi.xi(left,Hx,Hy,Hz);
    end
    criterion=norm(vec((left.*(xi.xi(right,Hx,Hy,Hz)))-p));
    if isnan(left(1)),
        error(['regularization too weak inside Simple Sinkhorn, computation of W bar derivative blew up iterations']);                   
    end                                             
end

val=sum(vec(safe_ulogu(left).*(xi.xi(right,Hx,Hy,Hz))));
val=val+sum(vec(xi.xi(left,Hx,Hy,Hz).*safe_ulogu(right))); % not a scaling by gamma here???
end