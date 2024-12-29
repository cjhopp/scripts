srcs = 24;
recs = 44;
load('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/TravelTimePicksFSC.mat')
winsizes = [3e-3,1e-2,3e-2];
for k = 1:length(winsizes)
    winsize  = winsizes(k);
    wintaper = 0.0003;
    
    for gath = 211
        disp(['Now working on ' num2str(gath) '.'])
        pick  = data(gath,1);
        if pick ~= 0
            t0 = pick + winsize/2 - wintaper;
            % [dsitempgath]                                           = dsiCASSMdelaySequenceEstWinPar(dsitempgath,t0,wintaper,winsize,84,dsitempgath.fh{8}/10,gath,1,1/10000);
            [dsitempgath.th{gath}(80+k,:),dsitempgath.th{gath}(90+k,:)] = dsi_rmsAmpWin(dsitempgath,t0,winsize,wintaper,gath);
            % dsitempgath.th{gath}(80+k,:) = dsitempgath.th{gath}(80+k,:) - mean(dsitempgath.th{gath}(80+k,:));
            [~,~,~,dsitempgath.th{gath}(100+k,:),~,~] = dsi_centfreqWin(dsitempgath,t0,winsize,wintaper,gath);
            % dsitempgath.th{gath}(85,:) = (pi .* dsitempgath.th{gath}(84,:) .* (dsitempgath.th{gath}(83,:).^2) .* dsitempgath.th{gath}(82,:) .* dsitempgath.th{gath}(83,1) .* sqrt(dsitempgath.th{gath}(82,1)/2)) ...
            %     ./ (2 .* (dsitempgath.th{gath}(82,:) .* (dsitempgath.th{gath}(83,:).^2) - dsitempgath.th{gath}(82,1) .* dsitempgath.th{gath}(83,1).^2));
        end

    end
end



% 
% figure(10)
%     subplot(2,1,1)
%         plot(dsitempgath.th{211}(81,:))
%     subplot(2,1,2)
%         plot(dsitempgath.th{211}(85,:))
% 
% figure
%     plot(dsitempgath.th{211}(82,:))


figure(10)
    subplot(3,1,1)
        plot(1:496,dsitempgath.th{211}(81:83,:))
        legend({'3ms window','10ms window','30ms window'})
        ylabel('RMS Amplitude')
    subplot(3,1,2)
        plot(1:496,dsitempgath.th{211}(91:93,:))
        ylabel('Peak Amplitude')
    subplot(3,1,3)
        plot(1:496,dsitempgath.th{211}(101:103,:))
        ylabel('Centroid Frequency')
        xlabel('Epoch')