close all;
clear all;
clc
resPath = './results/working/17_8/';
mkdir(resPath);
exnFile = fopen('exceptions.txt','a'); % exception file

%% load data, setting
data = load('filtered_data.mat');
mu = data.mu;
y = data.y1;
y = y/sqrt(var(y));
T = length(y);
Yori = y;
y = Yori+randn(T,1)/10;

y = y(8e3+(1:2e5));
Ycorrect = Yori(8e3+(1:2e5));
reRate = 4;
y = resample(y,1,reRate);
Ycorrect = resample(Ycorrect,1,reRate);
%figure, plot(y)
T = length(y);
%% setting missing blks
s = 80;
mInd = [400 1500 2400 3700 4800 9000 9400 ...
    1.16e4 1.3e4 1.7e4 1.8e4 2.02e4 2.14e4 ...
    2.85e4 2.9e4 3e4 3.55e4 3.8e4 3.96e4 ...
    4.52e4 4.6e4 4.72e4 4.78e4 4.83e4 4.87e4];
missingInd = zeros(T,1);
for m = 1:length(mInd);
    j = (mInd(m))+(1:randi(1)*s);
    j(j>T) = [];
    missingInd(j) = 1;
end
missingInd = missingInd==1;

Xtrain = 1:T;
YtestOri = y; YtestOri(missingInd) = 0;
YtrainOri = y;
missingTrain = missingInd;
YtrainOri(missingTrain) = 0;

%% init params based on the signal spectrum
noComps = 2;
window = 100;
overlap = 0.5;
params = initSECosParams(YtrainOri,data.fs*reRate,noComps,window,overlap);
%% training and prediction
% parameters
tau2vec = [2 5 10 20];
ratevec = [1/4 1/5 1/10 1/20 1/25];
MM = [16 32 64 128 256 512 1024 1500];

l1 = length(tau2vec);
l2 = length(ratevec);
lm = length(MM);
noTrials = 3;
noEvals = 200;
methods = {'local','chain','fitc','var'};

for k = 1:noTrials
    for m = 1:length(methods)
        method = methods{m};
        res = struct;
        if strcmpi(method,'chain')
            smse1 = zeros(l1,l2); % reconstruction error
            smse2 = zeros(l1,l2); % envelop reconstruction error
            msll = zeros(l1,l2);
            trainTime = zeros(l1,l2); % training time
            testTime = zeros(l1,l2); % test time
            hypers = cell(l1,l2); % hypers
            for i = 1:l1
                for j = 1:l2
                    disp(['running trial ' num2str(k) '/' num2str(noTrials) ', '...
                        'method ' method ', ' ...
                        'i = ' num2str(i) '/' num2str(l1) ', '...
                        'j = ' num2str(j) '/' num2str(l2)]);
                    tau2 = tau2vec(i);
                    rate = ratevec(j);
                    tau1 = tau2/rate;
                    K = floor(T/tau1);
                    Xtrain = 1:tau1*K;
                    Ytrain = YtrainOri(Xtrain);
                    Ytrue = Ycorrect(Xtrain);
                    Ynoisy = y(Xtrain);
                    Ytest = YtestOri(Xtrain);
                    missing = missingTrain(Xtrain);
                    missingStack = reshape(missing,[tau1,K])';
                    
                    covfunc = {@covSEiso};
                    theta_init = log(params);
                    try
                        tic
                        [theta_end,nlml] = testing_trainSECosMix(theta_init,noComps,covfunc,...
                            Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);
                        trainTime(i,j) = toc;
                        tic
                        [fest,vest] = testing_predictSECosMix(theta_end,noComps,covfunc,...
                            Xtrain',Ytest,tau1,tau2,missingStack);
                        testTime(i,j) = toc;
                        
                        % data reconstruction loss
                        ytrue = Ytrue(missing);
                        ynoisy = Ynoisy(missing);
                        yreco = fest(missing);
                        vreco = vest(missing);
                        smse1(i,j) = smsError(ytrue,yreco);
                        
                        % envelop reconstruction loss
                        env1 = abs(hilbert(y));
                        muEst = exp(theta_end(3));
                        env2 = abs(hilbert(fest));
                        etrue = env1(missing);
                        ereco = env2(missing);
                        smse2(i,j) = smsError(etrue,ereco);
                        
                        % msll
                        meanTrain =  mean(Ytrain);
                        varTrain = var(Ytrain);
                        msll(i,j) = mslLoss(ynoisy,yreco,vreco+exp(theta_end(end)),meanTrain,varTrain);
                        
                        hypers{i,j} = theta_end;
                    catch exception
                        disp(exception);
                        trainTime(i,j) = NaN;
                        testTime(i,j) = NaN;
                        smse1(i,j) = NaN;
                        smse2(i,j) = NaN;
                        msll(i,j) = NaN;
                        msg = [datestr(now) getReport(exception,'extended') '\n'];
                        fprintf(exnFile,msg);
                    end
                end
            end
            res.smse1 = smse1;
            res.smse2 = smse2;
            res.msll = msll;
            res.tau2vec = tau2vec;
            res.ratevec = ratevec;
            res.trainTime = trainTime;
            res.testTime = testTime;
            res.hypers = hypers;
            res.Ytrain = YtrainOri;
        elseif strcmpi(method,'local')
            smse1 = zeros(l1,l2); % reconstruction error
            smse2 = zeros(l1,l2); % envelop reconstruction error
            msll = zeros(l1,l2);
            trainTime = zeros(l1,l2); % training time
            testTime = zeros(l1,l2); % test time
            hypers = cell(l1,l2); % hypers
            for i = 1:l1
                for j = 1:l2
                    disp(['running trial ' num2str(k) '/' num2str(noTrials) ', '...
                        'method ' method ', ' ...
                        'i = ' num2str(i) '/' num2str(l1) ', '...
                        'j = ' num2str(j) '/' num2str(l2)]);
                    tau2 = tau2vec(i);
                    rate = ratevec(j);
                    tau1 = tau2/rate;
                    
                    K = floor(T/tau1);
                    Xtrain = 1:tau1*K;
                    Ytrain = YtrainOri(Xtrain);
                    Ytrue = Ycorrect(Xtrain);
                    Ynoisy = y(Xtrain);
                    Ytest = YtestOri(Xtrain);
                    missing = missingTrain(Xtrain);
                    missingStack = reshape(missing,[tau1,K])';
                    
                    covfunc = {@covSEiso};
                    theta_init = log(params);
                    try
                        tic
                        [theta_end,nlml] = testing_trainSECosMixLocal(theta_init,noComps,covfunc,...
                            Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);
                        trainTime(i,j) = toc;
                        tic
                        [fest,vest] = testing_predictSECosMixLocal(theta_end,noComps,covfunc,...
                            Xtrain',Ytest,tau1,tau2,missingStack);
                        testTime(i,j) = toc;
                        
                        % data reconstruction loss
                        ytrue = Ytrue(missing);
                        ynoisy = Ynoisy(missing);
                        yreco = fest(missing);
                        vreco = vest(missing);
                        smse1(i,j) = smsError(ytrue,yreco);
                        
                        % envelop reconstruction loss
                        env1 = abs(hilbert(y));
                        muEst = exp(theta_end(3));
                        env2 = abs(hilbert(fest));
                        etrue = env1(missing);
                        ereco = env2(missing);
                        smse2(i,j) = smsError(etrue,ereco);
                        
                        % msll
                        meanTrain =  mean(Ytrain);
                        varTrain = var(Ytrain);
                        msll(i,j) = mslLoss(ynoisy,yreco,vreco+exp(theta_end(end)),meanTrain,varTrain);
                        
                        hypers{i,j} = theta_end;
                    catch exception
                        disp(exception);
                        trainTime(i,j) = NaN;
                        testTime(i,j) = NaN;
                        smse1(i,j) = NaN;
                        smse2(i,j) = NaN;
                        msll(i,j) = NaN;
                        msg = [datestr(now) getReport(exception,'extended') '\n'];
                        fprintf(exnFile,msg);
                    end
                end
            end
            res.smse1 = smse1;
            res.smse2 = smse2;
            res.msll = msll;
            res.tau2vec = tau2vec;
            res.ratevec = ratevec;
            res.trainTime = trainTime;
            res.testTime = testTime;
            res.hypers = hypers;
            res.Ytrain = YtrainOri;
        elseif strcmpi(method,'fitc')
            smse1 = zeros(lm,1); % reconstruction error
            smse2 = zeros(lm,1); % envelop reconstruction error
            msll = zeros(lm,1);
            trainTime = zeros(lm,1); % training time
            testTime = zeros(lm,1); % test time
            hypers = cell(lm,1); % hypers
            XtrainFITC = Xtrain;
            XtrainFITC(missingTrain) = [];
            YtrainFITC = YtrainOri;
            YtrainFITC(missingTrain) = [];
            XtestFITC = Xtrain(missingTrain);
            Ytrue = Ycorrect(missingTrain);
            Ynoisy = y(missingTrain);
            for i = 1:lm
                disp(['running trial ' num2str(k) '/' num2str(noTrials) ', '...
                    'method ' method ', ' ...
                    'i = ' num2str(i) '/' num2str(lm)]);
                covfunc = {@covSum,{{@covProd,{@covSEisoU,@covCos}},...
                    {@covProd,{@covSEisoU,@covCos}}}};
                p = zeros(3*noComps,1);
                p(3*(0:noComps-1)+1) = params(2*(0:noComps-1)+1);
                p(3*(0:noComps-1)+2) = params(2*(0:noComps-1)+2);
                p(3*(0:noComps-1)+3) = params(2*noComps+(1:noComps));
                p(3*noComps+1) = params(3*noComps+1);
                                
                theta_init = log(p);
                try
                    tic
                    [theta_end,nlml] = trainSECosFITC(theta_init,covfunc,...
                        XtrainFITC',YtrainFITC,MM(i),noEvals);
                    trainTime(i) = toc;
                    tic
                    [fest,vest] = predictSECosFITC(theta_end,covfunc,XtrainFITC',...
                        YtrainFITC,XtestFITC',MM(i));
                    testTime(i) = toc;
                    % data reconstruction loss
                    smse1(i) = smsError(Ytrue,fest);
                    % envelop reconstruction loss
                    env1 = abs(hilbert(Ytrue));
                    env2 = abs(hilbert(fest));
                    smse2(i) = smsError(env1,env2);
                    % msll
                    meanTrain =  mean(YtrainFITC);
                    varTrain = var(YtrainFITC);
                    msll(i) = mslLoss(Ynoisy,fest,vest+exp(2*theta_end.lik),meanTrain,varTrain);
                    
                    hypers{i} = theta_end;
                catch exception
                    disp(exception);
                    trainTime(i) = NaN;
                    testTime(i) = NaN;
                    smse1(i) = NaN;
                    smse2(i) = NaN;
                    msll(i) = NaN;
                    msg = [datestr(now) getReport(exception,'extended') '\n'];
                    fprintf(exnFile,msg);
                end
            end
            
            res.smse1 = smse1;
            res.smse2 = smse2;
            res.msll = msll;
            res.M = MM;
            res.trainTime = trainTime;
            res.testTime = testTime;
            res.hypers = hypers;
            res.Ytrain = YtrainOri;
        elseif strcmpi(method,'var')
            smse1 = zeros(lm,1); % reconstruction error
            smse2 = zeros(lm,1); % envelop reconstruction error
            msll = zeros(lm,1);
            trainTime = zeros(lm,1); % training time
            testTime = zeros(lm,1); % test time
            hypers = cell(lm,1); % hypers
            XtrainVFE = Xtrain;
            XtrainVFE(missingTrain) = [];
            YtrainVFE = YtrainOri;
            YtrainVFE(missingTrain) = [];
            XtestVFE = Xtrain(missingTrain);
            Ytrue = Ycorrect(missingTrain);
            Ynoisy = y(missingTrain);
            for i = 1:lm
                disp(['running trial ' num2str(k) '/' num2str(noTrials) ', '...
                    'method ' method ', ' ...
                    'i = ' num2str(i) '/' num2str(lm)]);
                covfunc = {@covSum,{{@covProd,{@covSEisoU,@covCos}},...
                    {@covProd,{@covSEisoU,@covCos}}}};
                p = zeros(3*noComps,1);
                p(3*(0:noComps-1)+1) = params(2*(0:noComps-1)+1);
                p(3*(0:noComps-1)+2) = params(2*(0:noComps-1)+2);
                p(3*(0:noComps-1)+3) = params(2*noComps+(1:noComps));
                p(3*noComps+1) = params(3*noComps+1);
                                
                theta_init = log(p);
                try
                    [trainingTime,testTimei,hyp,nlml,my,vy] = ...
                        evalVarSE(XtrainVFE',YtrainVFE,XtestVFE',MM(i),theta_init,covfunc,noEvals);
                    
                    trainTime(i) = trainingTime;
                    testTime(i) = testTimei;
                    % data reconstruction loss
                    smse1(i) = smsError(Ytrue,my);
                    % envelop reconstruction loss
                    env1 = abs(hilbert(Ytrue));
                    env2 = abs(hilbert(my));
                    smse2(i) = smsError(env1,env2);
                    % msll
                    meanTrain =  mean(YtrainVFE);
                    varTrain = var(YtrainVFE);
                    msll(i) = mslLoss(Ynoisy,my,vy,meanTrain,varTrain);
                    
                    hypers{i} = hyp;
                catch exception
                    disp(exception);
                    trainTime(i) = NaN;
                    testTime(i) = NaN;
                    smse1(i) = NaN;
                    smse2(i) = NaN;
                    msll(i) = NaN;
                    msg = [datestr(now) getReport(exception,'extended') '\n'];
                    fprintf(exnFile,msg);
                end
            end
            
            res.smse1 = smse1;
            res.smse2 = smse2;
            res.msll = msll;
            res.M = MM;
            res.trainTime = trainTime;
            res.testTime = testTime;
            res.hypers = hypers;
            res.Ytrain = YtrainOri;
        end
        save([resPath 'result_SECos_two_bands_' method '_trial_' num2str(k) '.mat'],'res');
    end
end



