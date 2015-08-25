%% load data, setting
close all; clc; clear all;
fprintf('train and test on a toy example using SE covariance ...\n')
data = load('audio_subband.mat');
y = data.y;
T = length(y);
y = real(hilbert(y).*exp(-1i*2*pi*data.mu*[1:T]'));
yOri = y;
y = y+randn(T,1)/50; % add some noise to the original data

% only use first few samples for training and testing for now
y = y(8e3+(1:5e4));
y = resample(y,1,4);
yOri = yOri(8e3+(1:2e5));
yOri = resample(yOri,1,4);
figure, plot(y)
T = length(y);

%% setting missing blocks
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


%%
YtestOri = y; YtestOri(missingInd) = 0;
YtrainOri = y;
missingTrain = missingInd;
YtrainOri(missingTrain) = 0;

%% training and prediction

% parameters
noEvals = 200;

tau2 = 10;
rate = 1/10;
tau1 = tau2/rate;

K = floor(T/tau1);
Xtrain = 1:tau1*K;
Ytrain = YtrainOri(Xtrain);
Ytest = YtestOri(Xtrain);
missing = missingTrain(Xtrain);
missingStack = reshape(missing,[tau1,K])';

covfunc = {@covSEiso};
params = zeros(3,1);
params(1) = 50; % lengthscale
params(2) = sqrt(var(Ytrain)); % signal variance
params(3) = 1/2*sqrt(var(Ytrain)); % noise variance
theta_init = log(params);

[theta_end,nlml] = trainSE(theta_init,covfunc,...
    Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);

[fest,vest] = predictSE(theta_end,covfunc,...
    Xtrain',Ytest,tau1,tau2,missingStack);

% data reconstruction loss
ytrue = yOri(missing);
ynoisy = y(missing);
yreco = fest(missing);
vreco = vest(missing);

res = smsError(ytrue,yreco);
fprintf('smse %.3f\n', res);
meanTrain =  mean(Ytrain);
varTrain = var(Ytrain);
res = mslLoss(ynoisy,yreco,vreco+exp(theta_end(end)),meanTrain,varTrain);
fprintf('msll %.3f\n', res);

%% plot
x = (1:length(Ytrain))';
figure(1),
plot(x, Ytrain, '-r'), hold on;
plot(x, fest, '-b')
legend('train with missing bits', 'prediction')
plot(x, fest+2*sqrt(vest), '--b')
plot(x, fest-2*sqrt(vest), '--b')
hold off;
xlim([2000 6000])
xlabel('t'), ylabel('y'), title('zoom in to see error bars')
