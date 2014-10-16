% Experiment: inference speed/accuracy for various lengthscales, tau1, tau2
close all; clear all; clc
%% generate data
lengthscales = [1 2 5 10:10:100];
sn = sqrt(0.1); % previously 0.2, change to 0.1 on 15/7
noTrials = 3;
%tau2vec = 1:5:50;
%ratevec = 1./(20:-2:1);
tau2vec = [1 2 5 10];
ratevec = 5./(5:5:200);
LL = length(lengthscales);
LT = length(tau2vec);
LR = length(ratevec);

smse = zeros(noTrials,LL,LT,LR);
infTime = zeros(noTrials,LL,LT,LR);
N = 20000;
covfunc = {@covSEiso};

for l = 1:LL
    for k = 1:noTrials
        ell = lengthscales(l);
        y = sampleGPSE(1,ell,N);
        yn = y + sn*randn(N,1);
        x =(1:N)';        
        theta = [log(ell) 0 log(sn)];
        for i = 1:LT
            for j = 1:LR
                fprintf('%d/%d %d/%d %d/%d %d/%d\n',l,LL,k,noTrials,i,LT,j,LR)
                tau2 = tau2vec(i);
                tau1 = tau2/ratevec(j);
                ind = floor(N/tau1)*tau1;
                xtest = x(1:ind); ynoisy = yn(1:ind); ytrue = y(1:ind);
                fh = @() predictSE(theta,covfunc,xtest,ynoisy,tau1,tau2,[]);
                %tic
                [fest,vest] = feval(fh);
                %time1 = toc;
                smse(k,l,i,j) = smsError(ytrue,fest);
                infTime(k,l,i,j) = timeit(fh);
                %infTime(k,l,i,j) = time1;
            end
        end
    end
end

%% save results
res = struct();
res.smse = smse;
res.infTime = infTime;
res.lengthscales = lengthscales;
res.noTrials = noTrials;
res.tau2vec = tau2vec;
res.ratevec = ratevec;

resPath = './results/working/15_7/';
mkdir(resPath);
save([resPath 'results_ls_time_accu.mat'],'res');

