function [channel,enc]=decodeCASSMChannel(t, enc,threshold,fig)

% t         :  array of time values for the encoding data in ms
% enc       :  the CASSM encoding channel data
% threshold :  clips data below threshold (use .5)
% fig       :  plots results on figure(fig) id fig>0
%
% chamnel   :  the decoded 0 based Cytek channel number

 BYTE_SIZE=5;
 enc(find(enc<threshold*max(enc)))=0; 
 enc(find(enc>0))=max(enc); 
 fDiff=abs(diff(enc));

 
 b=t(find(fDiff>threshold*max(fDiff)));
 bit_width=round(b(2)-b(1));
 bit_start=b(1);
 
 data_start=bit_start+1.5*bit_width;
 bits=[];
 channel=0;
 for k=1:BYTE_SIZE
     bits=[bits,(data_start+k*bit_width)];
     if enc(min(find(t>=bits(k)))) > 0
        channel=bitset(channel,k);
     end
 end
 
 if fig
    figure(fig);
    clf;
    plot(t,enc);
    set(gca,'XLim',[floor(data_start-2*bit_width) data_start+8*bit_width]);
    hold;
    plot(t,[abs(diff(enc)); 0],'g-');
    plot(bits,ones(1,length(bits)) *threshold*max(enc),'r+');
    xlabel('Time');
    ylabel('Signal');
    text(bits(BYTE_SIZE)+bit_width,threshold*max(enc), ...
        num2str(channel),'fontsize',20)
 end

