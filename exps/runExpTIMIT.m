% 28/7: 10 components, 29/7: 20 components
close all;
clear all;
clc
resPath = './results/working/29_7/';
mkdir(resPath);
listfile = 'allfilelist.txt';
fid = fopen(listfile);
files = textscan(fid,'%s','delimiter','\n');
fclose(fid);
fdir = './datasets/audio/timit/';
SNRs = [-10 -5 0 5 10 15 20 25];
s = 1;
myformat = 'fname: %s old_snr: %.2f new_snr: %.2f\n';
fid = fopen([resPath 'result_set_' num2str(s) '.txt'],'a');
for i = 20*(s-1) + (1:20);
    fname = [fdir files{1}{i} '.wav'];
    [y,fs] = audioread(fname);
    % normalise signal
    rerate = 2;
    y = resample(y,1,2);
    y = y-mean(y);
    y = y/std(y);
    %noComps = 10;
    %window = 200;
    noComps = 20;
    window = 150;
    overlap = 0.5;
    params = initSECosParams(y,fs/rerate,noComps,window,overlap);
    %drawnow;
    %keyboard
    tau2 = 5;
    tau1 = 100;
    noEvals = 200;
    T = length(y);
    K = floor(T/tau1);
    X = (1:tau1*K)'; % ignore samples at the end for now
    Y = y(X);
    wnoise = randn(tau1*K,1);
    for j = 1:length(SNRs)
        fprintf('set %d %s %d/%d\n',s, fname,j,length(SNRs));
        Ynoisy = addnoise(Y,wnoise,SNRs(j));
        missing = zeros(size(X)) == 1;
        missingStack = reshape(missing,[tau1,K])';
        covfunc = {@covSEiso};
        theta_init = log(params);
        [theta_end,nlml] = testing_trainSECosMix(theta_init,noComps,covfunc,...
            X,Ynoisy,tau1,tau2,missingStack,noEvals);
        [fest,vest] = testing_predictSECosMix(theta_end,noComps,covfunc,...
            X,Ynoisy,tau1,tau2,missingStack);
        old_snr = SNRs(j);
        new_snr = snr(Y,Y-fest);
        fprintf(fid,myformat,fname,old_snr,new_snr);
    end
end
fclose(fid);