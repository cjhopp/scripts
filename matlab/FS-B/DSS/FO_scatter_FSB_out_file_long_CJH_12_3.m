clear all
close all
clc

%Import the data file
filename='/media/chet/data/chet-FS-B/DSS/FSB-SMF-3_Relative_Strain.txt';
T = readtable(filename, 'HeaderLines',42); 
T(:,end)=[];
A = table2array(T);

output_filename='/media/chet/data/chet-FS-B/DSS/maria_output/BFSB7_down.out';
%% CJH Maria's mappings based on foot-marks on fiber jacket
% BM_ent_B5=76.46; %Borehole mouth entrance [m] for BFS_B5
% BM_exit_B5=194.11;  
% 
% BM_ent_B3=232.21; %Borehole mouth entrance [m] for BFS_B3
% BM_exit_B3=401.37;
% 
% BM_ent_B4=406.56; %Borehole mouth entrance [m] for BFS_B4
% BM_exit_B4=566.58;
% 
% BM_ent_B6=588.22; %Borehole mouth entrance [m] for BFS_B6
% BM_exit_B6=688.19;
% 
% BM_ent_B7=693.37; %Borehole mouth entrance [m] for BFS_B7
% BM_exit_B7=787.86;
%% CJH Mappings from 'tug tests' by LBL staff

BM_ent_B5=80.97; %Borehole mouth entrance [m] for BFS_B5
BM_exit_B5=199.63;  

BM_ent_B3=237.7; %Borehole mouth entrance [m] for BFS_B3
BM_exit_B3=404.07;

BM_ent_B4=413.52; %Borehole mouth entrance [m] for BFS_B4
BM_exit_B4=571.90;

BM_ent_B6=594.76; %Borehole mouth entrance [m] for BFS_B6
BM_exit_B6=694.32;

BM_ent_B7=700.43; %Borehole mouth entrance [m] for BFS_B7
BM_exit_B7=793.47;

%%
%DOWN
B5_begin=2+BM_ent_B5; B5_end=2+BM_exit_B5; B5_end=(B5_begin+B5_end)/2;
B3_begin=2+BM_ent_B3; B3_end=2+BM_exit_B3; B3_end=(B3_begin+B3_end)/2;
B4_begin=2+BM_ent_B4; B4_end=2+BM_exit_B4; B4_end=(B4_begin+B4_end)/2;
B6_begin=2+BM_ent_B6; B6_end=2+BM_exit_B6; B6_end=(B6_begin+B6_end)/2;
B7_begin=2+BM_ent_B7; B7_end=2+BM_exit_B7; B7_end=(B7_begin+B7_end)/2;

% %UP
% B5_begin=2+BM_ent_B5; B5_end=2+BM_exit_B5; B5_begin=(B5_begin+B5_end)/2; 
% B3_begin=2+BM_ent_B3; B3_end=2+BM_exit_B3; B3_begin=(B3_begin+B3_end)/2; 
% B4_begin=2+BM_ent_B4; B4_end=2+BM_exit_B4; B4_begin=(B4_begin+B4_end)/2; 
% B6_begin=2+BM_ent_B6; B6_end=2+BM_exit_B6; B6_begin=(B6_begin+B6_end)/2; 
% B7_begin=2+BM_ent_B7; B7_end=2+BM_exit_B7; B7_begin=(B7_begin+B7_end)/2;

% CJH Here she's sorting out which integer channel is closest to the
% channel mapping (in meters) above
a=A(:,1);
[~,idx1]=min(abs(a-B5_begin)); minVal1=a(idx1);
[~,idx2]=min(abs(a-B5_end)); minVal2=a(idx2);
[~,idx3]=min(abs(a-B3_begin)); minVal1=a(idx3);
[~,idx4]=min(abs(a-B3_end)); minVal2=a(idx4);
[~,idx5]=min(abs(a-B4_begin)); minVal1=a(idx5);
[~,idx6]=min(abs(a-B4_end)); minVal2=a(idx6);
[~,idx7]=min(abs(a-B6_begin)); minVal1=a(idx7);
[~,idx8]=min(abs(a-B6_end)); minVal2=a(idx8);
[~,idx9]=min(abs(a-B7_begin)); minVal1=a(idx9);
[~,idx10]=min(abs(a-B7_end)); minVal2=a(idx10);
%%
ix=idx3; % index of the borehole: idx1 - B5, idx3 - B3, idx5 - B4, idx7 - B6, idx9 - B7; 
ix2=idx4; % index of the borehole: idx2 - B5, idx4 - B3, idx6 - B4, idx8 - B6, idx10 - B7;

B=A(ix:ix2,:);
% B=A;
% B(:,1)=B(:,1)-A(1); %for DOWN
B(:,1)=(B(:,1)-A(ix2))*(-1); %for UP
%%
depth=B(:,1)';

%
file = importdata(filename,''); 
Time=file(9,:);
Time = regexp(char(Time), '	', 'split'); Time(:,end)=[]; Time(:,1)=[];
Time=Time(:,196:498); % cut May 22nd 22/05/2019 16:35:11 to June 4th 10:14:20 
Time_p=datenum(Time);
Date=datestr(datenum(Time),'dd/mm/yyyy HH:MM:SS\t');
Time=mat2cell(Date,ones(length(Time),1),20);
Data=B(:,2:end);
Data=Data(:,196:498); % cut May 22nd 22/05/2019 16:35:11 to June 4th 10:14:20 
First_str = Data(:,end);
Data_relat= bsxfun(@minus,Data,First_str);

%%
figure
colormap(redblue(20))
surf(Time_p,depth,Data_relat,'FaceColor','interp',...
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
title(output_filename)
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
ylim([0 depth(end)]);
view(0,90)
set(gca,'Ydir','reverse')
grid on
colorbar
%%
%writing to file
dlmwrite(output_filename,'#DATE','delimiter','');
dlmwrite(output_filename,Time','delimiter','','-append');
dlmwrite(output_filename,'#','delimiter','','-append');

dlmwrite(output_filename,'#DEPTH','delimiter','','-append');
dlmwrite(output_filename,depth,'delimiter','\t','-append','precision','%.3f');
dlmwrite(output_filename,'#','delimiter','','-append');

dlmwrite(output_filename,'#DATA','delimiter','','-append');
dlmwrite(output_filename,Data_relat,'delimiter','\t','-append','precision','%.5f');
dlmwrite(output_filename,'#','delimiter','','-append');