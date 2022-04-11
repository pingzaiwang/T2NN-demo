function  F_T2NNUniformSampling4Videos(dataFolder, matName,maxFrames)

%% -------------DataSet-------------
matPath=[dataFolder   matName];
load(matPath);

if  (size(T, 3)>maxFrames)
    T=T(:,:,1:maxFrames);
end

%% ----------Global Settings----------
vObsRatio=[0.05, 0.1, 0.15];
vNSR=[0.05, 0.05, 0.05];

nSettings = length(vObsRatio);

for iS = 1:nSettings
    %% ---- Manually Set Paraeters ----
    obsRatio=vObsRatio(iS);
    NSR=vNSR(iS);
    %%----Manually Set Paraeters----
    
    %% ---Signal and Noise Generation---
    L = T/h_inf_norm(T);
    sz=size(L);   D=prod(sz);
    % ----------Process Tensor------------

    % Gaussian Noise
    sigma=NSR*h_tnorm(L)/sqrt(D);
    G=randn(sz)*sigma;
    
    % The Observed Tensor
    B = rand(sz)<obsRatio;
    vIdx=find(B>0);
    G = G.*B;
    Y=(L+G).*B;
    y=Y(vIdx);
    
    lamL=0.01*f_tensor_spectral_norm(G);

    gamma = 0.5;
    %gamma = 0.8;
    %alpha =[1 1 0.01];% [1 1 2]
    alpha =[1 1 100];% [1 1 2]
    %alpha =[10 1 1];% [1 1 2]
    alpha=alpha/sum(alpha);
    rho=1e-6; nu=1.1;
    
    % ---Observation---
    obs.tsize=sz;
    obs.y=y;
    obs.idx=vIdx;
    % ---Observation---
    
    optsOI.obs=obs;
    optsOI.para.lambda=lamL;
    optsOI.para.alpha=alpha';
    optsOI.para.gamma=gamma;
    optsOI.para.rho=rho;
    optsOI.para.nu=nu;
    optsOI.MAX_ITER_OUT=300;
    
    optsOI.MIN_RHO=1e-5;
    optsOI.MAX_RHO=1e5;
    optsOI.MAX_EPS=1e-5;
    optsOI.verbose=1;
    %-----construct memo-----
    memoOI=h_construct_memo(optsOI);
    memoOI.truth=L;
    memoOI.printerInterval=5;
    
    %------Ö´ÐÐ--------
    t=clock;
    memoOI=f_ntc_T2NN_admm(obs,optsOI,memoOI);
    t=etime(clock,t);
    %------Ö´ÐÐ--------
end

