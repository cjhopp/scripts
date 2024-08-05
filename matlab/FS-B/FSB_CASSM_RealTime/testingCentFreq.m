srcs = 24;
recs = 44;
load('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/TravelTimePicksFSC.mat')
winsize = 9e-3;
dt             = dsitempgath.fh{8};
samples = dsitempgath.fh{7};
wintaper = 0.0001;
    
% for gath = 1:recs*srcs
for gath = 211
    disp(['Now working on ' num2str(gath) '.'])
    pick  = data(gath,1);
    if pick ~= 0
        t0 = pick + winsize/2 - wintaper;

        % testing spectral ratio stuff
        [window,tvec,t1s,t2s] = cosWindow(dt,samples,t0,winsize,wintaper);
        % sub-sectioning base trace
        baseTrace             = dsitempgath.dat{gath}(:,1); 
        baseTraceW            = window.*baseTrace;
%             baseTraceWsub         = baseTraceW(t1s:t2s);

        % computing spectrum of base trace
        specbase = fftshift(fft(baseTraceW,4098));
        specbase = abs(specbase(2049:4098));
        f        = linspace(0,24000,length(specbase));
        for i = 1:dsitempgath.fh{1}
            % subsectioning monitor trace
            monTrace     = dsitempgath.dat{gath}(:,i);
            monTraceW    = window.*monTrace;
%                 monTraceWsub = monTraceW(t1s:t2s);

            specmon     = fftshift(fft(monTraceW,4098));
            specmon     = abs(specmon(2049:4098));

            specRat     = log(specbase./specmon);
            range       = 40:300;
            slope       = polyfit(f(range),specRat(range),1);

            dsitempgath.th{gath}(85,i) = slope(1);
            
            if i == 66
            figure(1)
                clf
                subplot(3,1,1)
                    hold on
                    taxis = linspace(0,dsitempgath.fh{10},dsitempgath.fh{7});
                    plot(taxis,baseTraceW)
                    plot(taxis,monTraceW)
                    hold off
                    xlabel('time samples')
                    legend({'base','monitor'})
                    ylabel('Trace Amp.')
                subplot(3,1,2)
                    hold on
                    plot(f,specbase)
                    plot(f,specmon)
                    xlim([0 5000])
                    ylabel('Frequency Spectrum')
                    hold off
                    xlabel('Hz')
                subplot(3,1,3)
                    hold on
                    plot(f,specRat)  
                    plot(f(range),slope(1)*f(range)+slope(2))
                    xlim([0 5000])
                    xlabel('Hz')
                    ylabel('Spectral Ratio (S.R.)')
                    legend({'base / monitor','best-fit line'})
            end
        end
    end
end

figure
    subplot(2,1,1)
        plotpercentchange(dsitempgath,gath,85,'k')
        ylabel('Slope of S.R.')
    subplot(2,1,2)
        plotpercentchange(dsitempgath,gath,81,'r')
        ylabel('RMS Atten.')
