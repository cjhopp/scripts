clear all
close all
clc

filename='BFSB5_down.out';

file = importdata(filename,''); 
Time=file(2,:);
Time = regexp(char(Time), '	', 'split'); Time(:,end)=[];
Time_p=datenum(Time,'dd/mm/yyyy HH:MM:SS\t');

Depth=file(5,:);
Depth = regexp(char(Depth), '	', 'split');
Depth=str2double(Depth);

Data=dlmread(filename,'',[7 0 length(Depth)-1+7 length(Time)-1]);

%%
figure
colormap(redblue(20))
surf(Time_p,Depth,Data,'FaceColor','interp',...
   'EdgeColor','none',...
   'FaceLighting','gouraud')
xlim([Time_p(end) Time_p(1)]);
newLim = get(gca,'XLim'); 
newx = linspace(newLim(1), newLim(2), 10); 
set(gca,'XTick', newx); 
datetick(gca,'x','dd-mmm','keepticks');
xlabel ('Date [day-month]')
ylabel ('Fiber length [m]')
zlabel('Relative strain [\mu\epsilon]')
title(filename)
grid minor
x0=10;
y0=10;
width=500;
height=250;
set(gcf,'units','points','position',[x0,y0,width,height])
set(gca,'FontSize',11);
xlabel ('Date [day-month]')
set(gca,'FontSize',9);
caxis([-50 50 ]);
ylim([0 Depth(end)]);
view(0,90)
set(gca,'Ydir','reverse')
grid on
colorbar


