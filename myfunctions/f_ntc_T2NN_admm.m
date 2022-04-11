function memo=f_ntc_T2NN_admm(obs,opts,memo)
% parameters
lambda=opts.para.lambda; 
gamma = opts.para.gamma;
alpha=opts.para.alpha; 
rho=opts.para.rho; 
nu=opts.para.nu;

% shortcuts
normTruth=norm(double(memo.truth(:)));
sz=obs.tsize;
K=length(sz);

% Set the observation positions 1
tMask=zeros(sz); tMask(obs.idx)=1; 
% observed tensor
tO=zeros(sz);  tO(obs.idx)=obs.y; 
% tensor variables X, 
tL=zeros(sz);
tK=zeros(sz);
tA=zeros(sz);
cT=cell(K,1);
cB=cell(K,1);
for k=1:K
    cT{k}=tL; 
    cB{k}=tL;
end

fprintf('++++f_ntc_T2NN_admm++++\n');
sz
for iter=1:opts.MAX_ITER_OUT
    oldL=tL; oldK = tK; oldCT=cT;
    
    tau=lambda*gamma/rho;
    tK = f_prox_TNN(tL - tA/rho,tau);
    
    tSumKT = tK;
    tSumAB = tA;
    for k=1:K
        M=f_unfold_k(tL-cB{k}/rho, k);
        tau=lambda*(1-gamma)*alpha(k)/rho;
        M=f_prox_NN(M,tau);
        cT{k}=f_fold_k(M,sz,k);
        tSumKT=tSumKT+cT{k};
        tSumAB=tSumAB+cB{k};
    end
    
    tL=(tMask.*tO + rho*tSumKT+tSumAB)./((K+1)*rho+tMask);
     
    
    % Stopping criteria
    eps = 0;
    eps = max(eps, f_inf_norm(tL-oldL));
    eps = max(eps, f_inf_norm(tK-oldK));
    for k=1:K
        eps = max(eps, f_inf_norm(cT{k}-oldCT{k}));
    end
    eps = max(eps, f_inf_norm(tL-tK));
    for k=1:K
        eps = max(eps,f_inf_norm(tL-cT{k}));
    end
    
    % Record 
    memo.iter=iter;
    memo.rho(iter)=rho; 
    memo.eps(iter)=eps;
    memo.err(iter)=norm(double( tL(:)-memo.truth(:) ))/normTruth;
    memo.pnsr(iter)=h_Psnr(memo.truth(:),tL(:));
    % Print iteration state
    
    if opts.verbose && mod(iter, memo.printerInterval)==0
    fprintf('++%d: PSNR=%2.3f, eps=%0.2e, err=%0.2e, rho=%0.2e\n', ...
                iter,memo.pnsr(iter),memo.eps(iter),memo.err(iter),memo.rho(iter));
    end
    
    if ( memo.eps(iter) <opts.MAX_EPS ) && ( iter > 10 )
        fprintf('Stopped:%d: PSNR=%2.3f, eps=%0.2e, err=%0.2e, rho=%0.2e\n', ...
                iter,memo.pnsr(iter),memo.eps(iter),memo.err(iter),memo.rho(iter));
        memo.T_hat=tL;
        break;
    end
    
    tA = tA + rho*(tK -tL);
    for k=1:K
        cB{k}=cB{k}+rho*( cT{k} - tL);
    end
    
    rho=min(rho*nu,opts.MAX_RHO);
end
memo.T_hat=tL;
