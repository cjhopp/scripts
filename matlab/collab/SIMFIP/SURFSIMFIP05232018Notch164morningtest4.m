% Matlab Routine for reading mHPP Data
% Implementation done by  Y. Guglielmi/J.Durand
%-----------------------------------------------18032015(JayVersion)forTournemire-----------------------------------------------
clear all
close all
clc

%----------SIMFIP----------
%----------Declaration----------
%
rep = '/home/chet/SIMFIP/SURF/SURF_Notch142ft/5-22-2018/';
ext = '*.csv';
chemin = fullfile(rep,ext)
list = dir(chemin)

TimeString=[];
TimeGString=[];
TimeD = [];
data0=[];

%All Test
n1=44;
n2=102;
% n2=102;
% n1=10;
% n2=90;

list(n1:n2).name

delimiter=',';
formatIn='mm/dd/yyyy HH:MM:SS.FFF';
ncol=85;
UsedColumns=linspace(2,86,85);
DeciBy = 100;
daysec=24*3600;
% 
for n = n1:n2
    
    file = fullfile(rep,list(n).name)
%    file = fullfile(rep1,'testfile2.dat')
    fid = fopen(file,'r');
    C = textscan(fid,['%s' repmat('%f',1,85)],'HeaderLines',1,'Delimiter',',');
    TimeString = [TimeString ; C{1}];
%    reltime = (time-time(1))*daysec
    if n==n1
       Time(1) = datenum(TimeString(1),formatIn);
    end
    if exist('bigdatab');
       bigdatab = [bigdatab ; C{UsedColumns}];
    else
       bigdatab = [C{UsedColumns}];
    end
%        linedata=linedata{1};
%        dataline = str2num(linedata(2:length(linedata)))
            
%        nextline = fgetl(fid);
    [nsamples nc]=size (bigdatab);
    if nsamples > 10000
        nsamples;
        datablock = [];
        for i=1:ncol
            datablock = [datablock decimate(bigdatab(:,i),DeciBy)];
        end
        data0 = [data0 ; datablock];
        clear bigdatab;
        
        Match = fliplr([length(TimeString):-DeciBy:1]);
        TimeGString = [TimeGString; TimeString(Match)];
        TimeD = datenum(TimeGString,formatIn);
        TimeString = [];
    
    end
 
     % SAUV_Hydro = [PetitDebitdeci,Pression_Injectiondeci,P_T_2_Pdeci];
end

    TimeSec = daysec*(TimeD-Time(1));


micTime = data0(:,1);
A1 = data0(:,2);
B1 = data0(:,3);
C1 = data0(:,4);
D1 = data0(:,5);
E1 = data0(:,6);
F1 = data0(:,7);
oneT1 = data0(:,8);
Pznm1 = data0(:,9);
Tznm1 = data0(:,10);%ok 
microSA1 = data0(:,11);
microSB1 = data0(:,12);
microSC1 = data0(:,13);
microSD1 = data0(:,14);
microSE1 = data0(:,15);
microSF1 = data0(:,16);
microS1T1 = data0(:,17);
microSPznm1 = data0(:,18);
microSTznm1 = data0(:,19);
Fox1 = data0(:,20);
Foy1 = data0(:,21);
Foz1 = data0(:,22);
Tox1 = data0(:,23);
Toy1 = data0(:,24);
Toz1 = data0(:,25);
Dx1 = data0(:,26);
Dy1 = data0(:,27);
Dz1 = data0(:,28);
Rx1 = data0(:,29);
% Ry1 = data0(:,28);
% Rz1 = data0(:,29);
T1 = data0(:,30);
Dxt1 = data0(:,31);
Dyt1 = data0(:,32);
Rxt1 = data0(:,33);
Ryt1 = data0(:,34);
Pz1RCf = data0(:,35);
Tz1 = data0(:,36);
Pbnm1 = data0(:,37);
Tbnm1 = data0(:,38);
Pb1 = data0(:,39);
Tb1 = data0(:,40);
A2 = data0(:,41);
B2 = data0(:,42);
C2 = data0(:,43);
D2 = data0(:,44);
E2 = data0(:,45);
F2 = data0(:,46);
OneT2 = data0(:,47);
Pznm2 = data0(:,48);
Tznm2 = data0(:,49);
microSA2 = data0(:,50);
microSB2 = data0(:,51);
microSC2 = data0(:,52);
microSD2 = data0(:,53);
microSE2 = data0(:,54);
microSF2 = data0(:,55);
microS1T2 = data0(:,56);
microSPznm2 = data0(:,57);
microSTznm2 = data0(:,58);
Fox2 = data0(:,59);
Foy2 = data0(:,60);
Foz2 = data0(:,61);
Tox2 = data0(:,62);
Toy2 = data0(:,63);
Toz2 = data0(:,64);
Dx2 = data0(:,65);
Dy2 = data0(:,66);
Dz2 = data0(:,67);
Rx2 = data0(:,68);
% Ry2 = data0(:,67);
% Rz2 = data0(:,68);
T2 = data0(:,69);
Dxt2 = data0(:,70);
Dyt2 = data0(:,71);
Rxt2 = data0(:,72);
Ryt2 = data0(:,73);
Pz2 = data0(:,74);
Tz2 = data0(:,75);
Pbnm2 = data0(:,76);
Tbnm2 = data0(:,77);
% Pb2 = data0(:,78);
% Tb2 = data0(:,79);
Up1 = data0(:,78);
Bp1 = data0(:,79);
Up2 = data0(:,80);
Bp2 = data0(:,81);
Pt1 = data0(:,82);
Pt2 = data0(:,83);
Flow = data0(:,84);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Enregistrement en .mat 
% % 
% SIMFIPSURF05222018brut10Hz = [TimeD,micTime,A1,B1,C1,D1,E1,F1,oneT1,Pznm1,Tznm1,microSA1,microSB1,microSC1,microSD1,microSE1,microSF1,microS1T1,microSPznm1,microSTznm1,Fox1,Foy1,Foz1,Tox1,Toy1,Toz1,Dx1,Dy1,Dz1,Rx1,T1,Dxt1,Dyt1,Rxt1,Ryt1,Pz1RCf,Tz1,Pbnm1,Tbnm1,Pb1,Tb1,A2,B2,C2,D2,E2,F2,OneT2,Pznm2,Tznm2,microSA2,microSB2,microSC2,microSD2,microSE2,microSF2,microS1T2,microSPznm2,microSTznm2,Fox2,Foy2,Foz2,Tox2,Toy2,Toz2,Dx2,Dy2,Dz2,Rx2,T2,Dxt2,Dyt2,Rxt2,Ryt2,Pz2,Tz2,Pbnm2,Tbnm2,Up1,Bp1,Up2,Bp2,Pt1,Pt2,Flow];
% save('SIMFIPSURF05222018brut10Hz.txt','SIMFIPSURF05222018brut10Hz', '-ASCII');
% save ('SIMFIPSURF05222018brut10Hz.mat','SIMFIPSURF05222018brut10Hz', '-ASCII');
% %
% stop
% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Processing data manually by Yves
% Correction effet de pression

% OneTT = [];
% OneTT1 = [];
% for i=1:19358
%   OneTT = 19.74;
% end
% OneTT1 = [OneTT1 ; OneTT];
% OneTT11 = 1.3 * OneTT1 + 0.89;
% Acor = A - OneTT11;
% Bcor = B - OneTT11;
% Cccor = Cc - OneTT11;
% Dcor = D - OneTT11;
% Ecor = E - OneTT11;
% Fcor = F - OneTT11;

% Calibration mA1: considering the transmission of the loads by the threads

mA1 = [0.021747883 -0.142064305 0.120457028 0.120410153 -0.14201743 0.021747883;
-0.151614577 -0.05694289 0.09461837 -0.09461837 0.056949289 0.151614577;
1.031246708 1.030927816 1.030927816 1.030927816 1.030927816 1.030927816;
0.019263011 0.038774139 0.019522408 -0.019522408 -0.038774139 -0.019263011;
-0.033663699 0.000146658 0.033505759 0.033517041 0.000146658 -0.033663699;
-0.005467588 0.005466153 -0.005467588 0.005467588 -0.005466153 0.005467588];

%  Calibration Co1: considering the transmission of the loads by the threads   
CO1 = [0.001196667 0 3.375e-07 0 0.0023775 0;
0 0.001197333 0 -0.0023725 0 -0.000128;
2.03333e-05 0 0.00001603 0 -0.00014125 0;
0 -1.2e-84489 0 0.001625984 0 -9.53402e-5;
1.2e-84744 0 7.56304e-07 0 0.001628276 0;
0 -7.63944e-7 0 -0.000139802 0 0.02293582];

Fx1 = A1*mA1(1,1)+B1*mA1(1,2)+C1*mA1(1,3)+D1*mA1(1,4)+E1*mA1(1,5)+F1*mA1(1,6);
Fy1 = A1*mA1(2,1)+B1*mA1(2,2)+C1*mA1(2,3)+D1*mA1(2,4)+E1*mA1(2,5)+F1*mA1(2,6);
Fz1 = A1*mA1(3,1)+B1*mA1(3,2)+C1*mA1(3,3)+D1*mA1(3,4)+E1*mA1(3,5)+F1*mA1(3,6);
Mx1 = A1*mA1(4,1)+B1*mA1(4,2)+C1*mA1(4,3)+D1*mA1(4,4)+E1*mA1(4,5)+F1*mA1(4,6);
My1 = A1*mA1(5,1)+B1*mA1(5,2)+C1*mA1(5,3)+D1*mA1(5,4)+E1*mA1(5,5)+F1*mA1(5,6);
Mz1 = A1*mA1(6,1)+B1*mA1(6,2)+C1*mA1(6,3)+D1*mA1(6,4)+E1*mA1(6,5)+F1*mA1(6,6);

ux1 = Fx1*CO1(1,1)+Fy1*CO1(1,2)+Fz1*CO1(1,3)+Mx1*CO1(1,4)+My1*CO1(1,5)+Mz1*CO1(1,6);
uy1 = Fx1*CO1(2,1)+Fy1*CO1(2,2)+Fz1*CO1(2,3)+Mx1*CO1(2,4)+My1*CO1(2,5)+Mz1*CO1(2,6);
uz1 = Fx1*CO1(3,1)+Fy1*CO1(3,2)+Fz1*CO1(3,3)+Mx1*CO1(3,4)+My1*CO1(3,5)+Mz1*CO1(3,6);
Tetax1 = Fx1*CO1(4,1)+Fy1*CO1(4,2)+Fz1*CO1(4,3)+Mx1*CO1(4,4)+My1*CO1(4,5)+Mz1*CO1(4,6);
Tetay1 = Fx1*CO1(5,1)+Fy1*CO1(5,2)+Fz1*CO1(5,3)+Mx1*CO1(5,4)+My1*CO1(5,5)+Mz1*CO1(5,6);
Tetaz1 = Fx1*CO1(6,1)+Fy1*CO1(6,2)+Fz1*CO1(6,3)+Mx1*CO1(6,4)+My1*CO1(6,5)+Mz1*CO1(6,6);

% Fx1cor = Acor*A1(1,1)+Bcor*A1(1,2)+Cccor*A1(1,3)+Dcor*A1(1,4)+Ecor*A1(1,5)+Fcor*A1(1,6);
% Fy1cor = Acor*A1(2,1)+Bcor*A1(2,2)+Cccor*A1(2,3)+Dcor*A1(2,4)+Ecor*A1(2,5)+Fcor*A1(2,6);
% Fz1cor = Acor*A1(3,1)+Bcor*A1(3,2)+Cccor*A1(3,3)+Dcor*A1(3,4)+Ecor*A1(3,5)+Fcor*A1(3,6);
% Mx1cor = Acor*A1(4,1)+Bcor*A1(4,2)+Cccor*A1(4,3)+Dcor*A1(4,4)+Ecor*A1(4,5)+Fcor*A1(4,6);
% My1cor = Acor*A1(5,1)+Bcor*A1(5,2)+Cccor*A1(5,3)+Dcor*A1(5,4)+Ecor*A1(5,5)+Fcor*A1(5,6);
% Mz1cor = Acor*A1(6,1)+Bcor*A1(6,2)+Cccor*A1(6,3)+Dcor*A1(6,4)+Ecor*A1(6,5)+Fcor*A1(6,6);
% 
% ux1cor = Fx1cor*CO1(1,1)+Fy1cor*CO1(1,2)+Fz1cor*CO1(1,3)+Mx1cor*CO1(1,4)+My1cor*CO1(1,5)+Mz1cor*CO1(1,6);
% uy1cor = Fx1cor*CO1(2,1)+Fy1cor*CO1(2,2)+Fz1cor*CO1(2,3)+Mx1cor*CO1(2,4)+My1cor*CO1(2,5)+Mz1cor*CO1(2,6);
% uz1cor = Fx1cor*CO1(3,1)+Fy1cor*CO1(3,2)+Fz1cor*CO1(3,3)+Mx1cor*CO1(3,4)+My1cor*CO1(3,5)+Mz1cor*CO1(3,6);
% Tetax1cor = Fx1cor*CO1(4,1)+Fy1cor*CO1(4,2)+Fz1cor*CO1(4,3)+Mx1cor*CO1(4,4)+My1cor*CO1(4,5)+Mz1cor*CO1(4,6);
% Tetay1cor = Fx1cor*CO1(5,1)+Fy1cor*CO1(5,2)+Fz1cor*CO1(5,3)+Mx1cor*CO1(5,4)+My1cor*CO1(5,5)+Mz1cor*CO1(5,6);
% Tetaz1cor = Fx1cor*CO1(6,1)+Fy1cor*CO1(6,2)+Fz1cor*CO1(6,3)+Mx1cor*CO1(6,4)+My1cor*CO1(6,5)+Mz1cor*CO1(6,6);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sonde Stimulation - Rotation des coordonnees Oz axe forage, Oy vertical, Ox vers Yates 

LS1 = length(ux1);

angS1X = pi*(0/180); % The angle (0 < angX < 360� )is around Ox  
angS1Y = pi*(0/180); % The angle (0 < angY < 360� )is around Oy  
angS1Z = pi*(-49/180); % The angle (0 < angZ < 360� )is around Oz  
% angS1Z = 50 ; % The angle (0 < angZ < 360� )is around Oz
% rS1x = [1,0,0;0,cos(angS1X),-sin(angS1X);0,sin(angS1X),cos(angS1X)];% Trigonometric (anticlockwise) rotation around the X axis ("rotation propre")
% rS1y = [cos(angS1Y),0,sin(angS1Y);0,1,0;-sin(angS1Y),0,cos(angS1Y)];% Trigonometric (anticlockwise) rotation around the Y axis ("Nutation")
rS1z = [cos(angS1Z),-sin(angS1Z),0;sin(angS1Z),cos(angS1Z),0;0,0,1];% Trigonometric (anticlockwise) rotation around the Z axis ("Precession")

calculS1 = [];
S1Yates = [];
S1Top = [];
S1WellAxial = [];

for ni = 1:LS1
calculS1 =  [ux1(ni,1);uy1(ni,1);uz1(ni,1)];  
rotatS1 = rS1z*calculS1;
S1Yates(ni,1) = rotatS1(1,1);
S1Top(ni,1) = rotatS1(2,1);
S1WellAxial(ni,1) = rotatS1(3,1);
end

% correction clamps

calculS1c = [];
S1Yatesc = [];
S1Topc = [];
S1WellAxialc = [];

Pz1RCfc = -7.2e-11*Up1(1:82721).^4+8.6e-7*Up1(1:82721).^3-0.0038*Up1(1:82721).^2+8.5*Up1(1:82721)-5.7e3-71-136-32-2;
Pz1RCfRC = cat(1,Pz1RCf(1:22152),Pz1RCfc(22153:23593),Pz1RCf(23594:27824),Pz1RCfc(27825:29226)*1+8,Pz1RCf(29227:82721));
Pz1RCfRCf = medfilt1(Pz1RCfRC,21);

for ni = 1:LS1
% calculS1c =  [ux1(ni,1)-(-0.00025719+(-17.2e-6*Pz1RCf(ni,1))/300);uy1(ni,1)-(-0.00039795+(23.0e-6*Pz1RCf(ni,1))/300);uz1(ni,1)-(0.000027204+(-6.01e-6*Pz1RCf(ni,1))/300)];  
calculS1c = [-1.05e-4*Pz1RCfRCf(ni,1)/300;5.84e-5*Pz1RCfRCf(ni,1)/300;-2.93e-6*Pz1RCfRCf(ni,1)/300];%good

rotatS1c = rS1z*calculS1c;
S1Yatesc(ni,1) = rotatS1c(1,1);
S1Topc(ni,1) = rotatS1c(2,1);
S1WellAxialc(ni,1) = rotatS1c(3,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Matlab Routine for reading PNNL flow/Pressure Data
%
rep2 = 'E:\PNNL_flowdata\5-23-2018PNNL';
ext2 = '*.csv';
chemin2 = fullfile(rep2,ext2);
list2 = dir(chemin2);

TimeString2=[];
TimeGString2=[];
TimeD2 = [];
data2=[];

%All Test
N1=1;
N2=2;

list(N1:N2).name

delimiter=',';
formatIn='mm/dd/yyyy HH:MM:SS';
ncol2=4;
UsedColumns2=linspace(2,5,4);
DeciBy2 = 1;
daysec2=24*3600;
% 
for n = N1:N2
    
    file2 = fullfile(rep2,list2(n).name)
%    file = fullfile(rep1,'testfile2.dat')
    fid = fopen(file2,'r');
    C2 = textscan(fid,['%s' repmat('%f',1,4)],'HeaderLines',3,'Delimiter',',');
    TimeString2 = [TimeString2 ; C2{1}];
%    reltime = (time-time(1))*daysec
    if n==N1
       Time2(1) = datenum(TimeString2(1),formatIn);
    end
    if exist('bigdatab2');
       bigdatab2 = [bigdatab2 ; C2{UsedColumns2}];
    else
       bigdatab2 = [C2{UsedColumns2}];
    end
%        linedata=linedata{1};
%        dataline = str2num(linedata(2:length(linedata)))
            
%        nextline = fgetl(fid);
    [nsamples2 nc]=size (bigdatab2);
    if nsamples2 > 100
        nsamples2
        datablock = [];
        for i=1:ncol2
            datablock = [datablock decimate(bigdatab2(:,i),DeciBy2)];
        end
        data2 = [data2 ; datablock];
        clear bigdatab2;
        
        Match2 = fliplr([length(TimeString2):-DeciBy2:1]);
        TimeGString2 = [TimeGString2; TimeString2(Match2)];
        TimeD2 = datenum(TimeGString2,formatIn)-6/24;
%         TimeD2.TimeZone = 'America/Rapid_City';
        TimeString2 = [];
    
    end
 
     % SAUV_Hydro = [PetitDebitdeci,Pression_Injectiondeci,P_T_2_Pdeci];
end

    TimeSec2 = daysec*(TimeD2-Time2(1));


% micTime = data2(:,1);
QuizixP = data2(:,1);
QuizixQ = data2(:,2);
QuizixV = data2(:,3);
TriplexQ = data2(:,4);

%% Enregistrement en .mat et en .txt
% 
% SIMFIPSURF05212018 = [A1(),B1(),C1(),D1(),E1(),F1(),OneT1(),microSA(),microSB(),microSC(),microSD(),microSE(),microSF(),microSOneT(),Fx(),Fy(),Fz(),Tx(),Ty(),Tz(),Dx(),Dy(),Dz(),Rx(),Ry(),Rz()];
% save('SIMFIPSURF05212018.txt','SIMFIPSURF05212018', '-ASCII');
% save ('SIMFIPSURF05212018.mat');
% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reconstitution de la pression de chambre au pic

figure (1)
% Pz1RCfc = -7.2e-11*Up1(1:82721).^4+8.6e-7*Up1(1:82721).^3-0.0038*Up1(1:82721).^2+8.5*Up1(1:82721)-5.7e3-71-136-32-2;
% Pz1RCfRC = cat(1,Pz1RCf(1:22152),Pz1RCfc(22153:23593),Pz1RCf(23594:27824),Pz1RCfc(27825:29226)*1+8,Pz1RCf(29227:82721));
% Pz1RCfRCf = medfilt1(Pz1RCfRC,21);
%plot(TimeD,Flow,'k-','linewidth',2);hold on
% plot(TimeD(14344:82719),Pz1RCfc(14344:82719),'r-','linewidth',2);hold on
plot(TimeD(1:82719),Pz1RCf(1:82719),'b-','linewidth',2);hold on
plot(TimeD(:),Pz1RCfRCf(:),'gr-','linewidth',2);hold on
plot(TimeD(14344:82719),Up1(14344:82719),'r-','linewidth',2);hold on
% plot(TimeD(14344:82719),Bp1(14344:82719),'gr-','linewidth',2);hold on
grid on;
NumTicks = 50;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
xlim([737203.478 737204.005]);
% ylim([-30 6000]);
%xlabel('Date');
ylabel('Pressure (psi)');
legend('Interval Pressure','Top Packer pressure','Bottom Packer Pressure');
% title ('Reference sensor');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure (2)
subplot(3,1,1)
%plot(TimeD,Flow,'k-','linewidth',2);hold on
plot(TimeD2,QuizixQ,'b-','linewidth',2);hold on
grid on;
NumTicks = 50;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
xlim([737203.478 737204.005]);
ylim([-0.01 0.41]);
%xlabel('Date');
ylabel('Flow (mL/min)');
title ('Fracture Propagation to 5 meters on May 22, 2018');

subplot(3,1,2)
%plot(TimeD,Flow,'k-','linewidth',2);hold on
plot(TimeD,Pz1RCfRCf,'b-','linewidth',2);hold on
plot(TimeD,Up1,'r-','linewidth',2);hold on
plot(TimeD,Bp1,'gr-','linewidth',2);hold on
grid on;
NumTicks = 50;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
xlim([737203.478 737204.005]);
ylim([-30 4500]);
%xlabel('Date');
ylabel('Pressure (psi)');
legend('Interval Pressure','Top Packer pressure','Bottom Packer Pressure');
% title ('Reference sensor');

%figure (2)
subplot(3,1,3)
plot(TimeD,S1Yates()-S1Yates(1)-(S1Yatesc()-S1Yatesc(1)-(0.0006617+0.0006364)),'b-','linewidth',2);hold on
plot(TimeD,S1Top()-S1Top(1)-(S1Topc()-S1Topc(1)-(2.1e-797+0.0009821)),'r-','linewidth',2);hold on
plot(TimeD,4*(S1WellAxial()-S1WellAxial(1)-(S1WellAxialc()-S1WellAxialc(1)-(1.719e-5+4.593e-6))),'gr-','linewidth',2);hold on

NumTicks = 50;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737203.478 737204.005]);
ylim([-5e-4 12e-4]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('X-Yates','Y-Top','10*Z-Well Axial');
% title ('Reference sensor');

figure (20)

 subplot(2,1,1)
%plot(TimeD,Flow,'k-','linewidth',2);hold on
plot(TimeD2,QuizixQ,'b-','linewidth',2);hold on
grid on;
NumTicks = 50;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
% xlim([737203.478 737204.005]);
ylim([-0.01 0.41]);
%xlabel('Date');
ylabel('Flow (mL/min)');
title ('Fracture Propagation to 5 meters on May 22, 2018');

 subplot(2,1,2)
%plot(TimeD,Flow,'k-','linewidth',2);hold on
plot(TimeD2,QuizixP,'b-','linewidth',2);hold on
plot(TimeD2,QuizixP,'c:','linewidth',2);hold on
grid on;
NumTicks = 50;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
% xlim([737203.478 737204.005]);
% ylim([-0.01 0.41]);
%xlabel('Date');
ylabel('Pressure (psi)');
% title ('Fracture Propagation to 5 meters on May 22, 2018');

% figure(5)
% 
% subplot(3,1,1)
% % plot(TimeD,Dx1()-Dx1(1),'b--','linewidth',2);hold on
% plot(TimeD,S1Yates()-S1Yates(1),'b-','linewidth',2);hold on
% plot(TimeD,S1Yatesc()-S1Yatesc(1)-(0.0006617+0.0006364),'b:','linewidth',2);hold on
% 
% NumTicks = 25;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
% ylabel('Displacement (micron)');
% legend('X-Yates','X-Yates c ');
% % title ('Reference sensor');
% 
% subplot(3,1,2)
% 
% % plot(TimeD,Dy1()-Dy1(1),'r--','linewidth',2);hold on
% plot(TimeD,S1Top()-S1Top(1),'r-','linewidth',2);hold on
% plot(TimeD,S1Topc()-S1Topc(1)-(2.1e-797+0.0009821),'r:','linewidth',2);hold on
% 
% NumTicks = 25;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
% ylabel('Displacement (micron)');
% legend('Y-Top','Y-Top c ');
% % title ('Reference sensor');
% 
% subplot(3,1,3)
% 
% % plot(TimeD,Dz1()-Dz1(1),'gr--','linewidth',2);hold on
% plot(TimeD,S1WellAxial()-S1WellAxial(1),'gr-','linewidth',2);hold on
% plot(TimeD,S1WellAxialc()-S1WellAxialc(1)-(1.719e-5+4.593e-6),'gr:','linewidth',2);hold on
% NumTicks = 25;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
% ylabel('Displacement (micron)');
% legend('Z-Well Axial','Z-Well Axial c ');
% % title ('Reference sensor');
% 
% % stop
% 
% figure(6)
% 
% subplot(3,1,1)
% 
% plot(TimeD,S1Yates()-S1Yates(1)-(S1Yatesc()-S1Yatesc(1)-(0.0006617+0.0006364)),'b-','linewidth',2);hold on
% 
% NumTicks = 25;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
% ylabel('Displacement (micron)');
% % legend('X-Yates','X-Yates c ');
% % title ('Reference sensor');
% 
% subplot(3,1,2)
% 
% plot(TimeD,S1Top()-S1Top(1)-(S1Topc()-S1Topc(1)-(2.1e-797+0.0009821)),'r-','linewidth',2);hold on
% 
% NumTicks = 25;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
% ylabel('Displacement (micron)');
% % legend('Y-Top','Y-Top c ');
% % title ('Reference sensor');
% 
% subplot(3,1,3)
% 
% 
% plot(TimeD,S1WellAxial()-S1WellAxial(1)-(S1WellAxialc()-S1WellAxialc(1)-(1.719e-5+4.593e-6)),'gr-','linewidth',2);hold on
% 
% NumTicks = 25;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
% ylabel('Displacement (micron)');
% % legend('Z-Well Axial','Z-Well Axial c ');
% % title ('Reference sensor');
% 
% % stop

% figure(7)
% 
% subplot(3,1,1)
% 
% plot(TimeD,Flow,'b-','linewidth',2);hold on
% NumTicks = 50;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
%  ylabel('Flowrate (mL/min)');
% % legend('X-Yates','X-Yates c ');
%  title ('Displacements corrected from Clamp effect');
% 
% subplot(3,1,2)
% 
% plot(TimeD,Pz1RCf,'b-','linewidth',2);hold on
% plot(TimeD,Up1,'r-','linewidth',2);hold on
% plot(TimeD,Bp1,'gr-','linewidth',2);hold on
% NumTicks = 50;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% ylim([900 4000]);
% grid on;
% %xlabel('Date');
% ylabel('Pressure (psi)');
% legend('Chamber','Top Packer','Bottom Packer');
% % title ('Reference sensor');
% 
% subplot(3,1,3)
% 
% plot(TimeD,S1Yates()-S1Yates(1)-(S1Yatesc()-S1Yatesc(1)-(0.0006617+0.0006364)),'b-','linewidth',2);hold on
% plot(TimeD,S1Top()-S1Top(1)-(S1Topc()-S1Topc(1)-(2.1e-797+0.0009821)),'r-','linewidth',2);hold on
% plot(TimeD,10*(S1WellAxial()-S1WellAxial(1)-(S1WellAxialc()-S1WellAxialc(1)-(1.719e-5+4.593e-6))),'gr-','linewidth',2);hold on
% 
% NumTicks = 50;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737203.485 737203.6]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
% ylabel('Displacement (micron)');
% legend('X-Yates','Y-Top','10*Z-Well Axial');
% % title ('Reference sensor');
% 
% %  stop

figure(71)

% subplot(1,2,1)
yyaxis left
plot(TimeD(12861:14572),(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726)),'k-','linewidth',3);hold on
plot(TimeD(14572:22108),(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726)),'c-','linewidth',3);hold on
plot(TimeD(22108:37769),(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726)),'r-','linewidth',3);hold on
plot(TimeD(37769:48598),(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726)),'b-','linewidth',3);hold on

yyaxis right
plot(TimeD(12861:48598),S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)),'b-','linewidth',2);hold on
plot(TimeD(12861:48598),S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)),'r-','linewidth',2);hold on
plot(TimeD(12861:48598),4*(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))),'gr-','linewidth',2);hold on

NumTicks = 20;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.5003896169 737202.518]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
% legend('Pressure (psi)','X-Yates','Y-Top','Z-Well Axial');
% legend('Chamber','Top Packer','Bottom Packer');
% title ('Pressure psi');

figure(710)

% subplot(1,2,1)
yyaxis left
plot(TimeD(12861:14572),(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726)),'k-','linewidth',3);hold on
plot(TimeD(14572:22108),(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726)),'c-','linewidth',3);hold on
plot(TimeD(22108:37769),(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726)),'r-','linewidth',3);hold on
plot(TimeD(37769:48598),(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726)),'b-','linewidth',3);hold on

yyaxis right
plot(TimeD(12861:48598),Tetax1(12861:48598)-Tetax1(12861),'b-','linewidth',2);hold on
plot(TimeD(12861:48598),Tetay1(12861:48598)-Tetay1(12861),'r-','linewidth',2);hold on
plot(TimeD(12861:48598),Tetaz1(12861:48598)-Tetaz1(12861),'gr-','linewidth',2);hold on

NumTicks = 20;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.5003896169 737202.518]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
legend('Pressure (psi)','Tetax','Tetay','Tetaz');
% legend('Chamber','Top Packer','Bottom Packer');
% title ('Pressure psi');

figure(72)

plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1Yates(12861:14572)-S1Yates(12861)-(S1Yatesc(12861:14572)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'k-','linewidth',3);hold on
plot(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726),(S1Yates(14572:22108)-S1Yates(12861)-(S1Yatesc(14572:22108)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'c-','linewidth',3);hold on
plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Yates(22108:37769)-S1Yates(12861)-(S1Yatesc(22108:37769)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'r-','linewidth',3);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Yates(37769:48598)-S1Yates(12861)-(S1Yatesc(37769:48598)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'b-','linewidth',3);hold on

plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1Top(12861:14572)-S1Top(12861)-(S1Topc(12861:14572)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'k:','linewidth',2);hold on
plot(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726),(S1Top(14572:22108)-S1Top(12861)-(S1Topc(14572:22108)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'c:','linewidth',2);hold on
plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Top(22108:37769)-S1Top(12861)-(S1Topc(22108:37769)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'r:','linewidth',2);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Top(37769:48598)-S1Top(12861)-(S1Topc(37769:48598)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'b:','linewidth',2);hold on

plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1WellAxial(12861:14572)-S1WellAxial(12861)-(S1WellAxialc(12861:14572)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861)))),'k-','linewidth',1);hold on
plot(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726),(S1WellAxial(14572:22108)-S1WellAxial(12861)-(S1WellAxialc(14572:22108)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861)))),'c-','linewidth',1);hold on
plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1WellAxial(22108:37769)-S1WellAxial(12861)-(S1WellAxialc(22108:37769)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861)))),'r-','linewidth',1);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxial(37769:48598)-S1WellAxial(12861)-(S1WellAxialc(37769:48598)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-(S1WellAxialc(12861))))),'b-','linewidth',1);hold on

xlim([500 4000]);
ylim([-1.1 1.1]);
grid on;
xlabel('Chamber Pressure psi');
ylabel('Normalized Displacements');
% title ('Pressure psi');

figure(73)

plot(QuizixP(1:6006),QuizixQ(1:6006),'b-','linewidth',2);hold on
xlim([500 4000]);
grid on
xlabel('Chamber Pressure psi');
ylabel('Flowrate (mL/Min)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% filtrage de la partie elastique de la chambre + probe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(74)

% plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1Yates(12861:14572)-S1Yates(12861)-(S1Yatesc(12861:14572)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'k-','linewidth',3);hold on
plot(Pz1RCfRCf(14713:17447)-Pz1RCfRCf(726),(S1Yates(14713:17447)-S1Yates(12861)-(S1Yatesc(14713:17447)-S1Yatesc(12861))),'c-','linewidth',3);hold on
% plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Yates(22108:37769)-S1Yates(12861)-(S1Yatesc(22108:37769)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'r-','linewidth',3);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Yates(37769:48598)-S1Yates(12861)-(S1Yatesc(37769:48598)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'b-','linewidth',3);hold on

% plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1Top(12861:14572)-S1Top(12861)-(S1Topc(12861:14572)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'k:','linewidth',2);hold on
plot(Pz1RCfRCf(14713:17447)-Pz1RCfRCf(726),(S1Top(14713:17447)-S1Top(12861)-(S1Topc(14713:17447)-S1Topc(12861))),'c:','linewidth',2);hold on
% plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Top(22108:37769)-S1Top(12861)-(S1Topc(22108:37769)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'r:','linewidth',2);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Top(37769:48598)-S1Top(12861)-(S1Topc(37769:48598)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'b:','linewidth',2);hold on

% plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1WellAxial(12861:14572)-S1WellAxial(12861)-(S1WellAxialc(12861:14572)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861)))),'k-','linewidth',1);hold on
plot(Pz1RCfRCf(14713:17447)-Pz1RCfRCf(726),(S1WellAxial(14713:17447)-S1WellAxial(12861)-(S1WellAxialc(14713:17447)-S1WellAxialc(12861))),'c-','linewidth',1);hold on
% plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1WellAxial(22108:37769)-S1WellAxial(12861)-(S1WellAxialc(22108:37769)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861)))),'r-','linewidth',1);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxial(37769:48598)-S1WellAxial(12861)-(S1WellAxialc(37769:48598)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-(S1WellAxialc(12861))))),'b-','linewidth',1);hold on

xlim([500 4000]);
% ylim([-1.1 1.1]);
grid on;
xlabel('Chamber Pressure psi');
ylabel('Normalized Displacements');
% title ('Pressure psi')

figure(741)


plot(Pz1RCfRCf(12861:37769)-Pz1RCfRCf(726),sqrt((S1Yates(12861:37769)-S1Yates(12861)-(S1Yatesc(12861:37769)-S1Yatesc(12861))).^2+(S1Top(12861:37769)-S1Top(12861)-(S1Topc(12861:37769)-S1Topc(12861))).^2),'c:','linewidth',3);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),sqrt((S1Yates(37769:48598)-S1Yates(12861)-(S1Yatesc(37769:48598)-S1Yatesc(12861))).^2+(S1Top(37769:48598)-S1Top(12861)-(S1Topc(37769:48598)-S1Topc(12861))).^2),'r:','linewidth',3);hold on

plot(Pz1RCfRCf(12861:37769)-Pz1RCfRCf(726),(S1WellAxial(12861:37769)-S1WellAxial(12861)-(S1WellAxialc(12861:37769)-S1WellAxialc(12861))),'c-','linewidth',1);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxial(37769:48598)-S1WellAxial(12861)-(S1WellAxialc(37769:48598)-S1WellAxialc(12861))),'r-','linewidth',1);hold on

xlim([500 4000]);
% ylim([-1.1 1.1]);
grid on;
xlabel('Chamber Pressure psi');
ylabel('Displacements');
legend('Radial',' ','Axial',' ');
% title ('Pressure psi')

%stop

figure(75)

plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1Yates(12861:14572)-S1Yates(12861)-(S1Yatesc(12861:14572)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(12861:14572)+7.8e-5))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(12861:48598)+7.8e-5))),'b-','linewidth',3);hold on
plot(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726),(S1Yates(14572:22108)-S1Yates(12861)-(S1Yatesc(14572:22108)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(14572:22108)+7.8e-5))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(12861:48598)+7.8e-5))),'b-','linewidth',3);hold on
plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Yates(22108:37769)-S1Yates(12861)-(S1Yatesc(22108:37769)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(22108:37769)+7.8e-5))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(12861:48598)+7.8e-5))),'b-','linewidth',3);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Yates(37769:48598)-S1Yates(12861)-(S1Yatesc(37769:48598)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(37769:48598)+7.8e-5))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(12861:48598)+7.8e-5))),'b-','linewidth',3);hold on

plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1Top(12861:14572)-S1Top(12861)-(S1Topc(12861:14572)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(12861:14572)+0.00019))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(12861:48598)+0.00019))),'r:','linewidth',2);hold on
plot(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726),(S1Top(14572:22108)-S1Top(12861)-(S1Topc(14572:22108)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(14572:22108)+0.00019))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(12861:48598)+0.00019))),'r:','linewidth',2);hold on
plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Top(22108:37769)-S1Top(12861)-(S1Topc(22108:37769)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(22108:37769)+0.00019))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(12861:48598)+0.00019))),'r:','linewidth',2);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Top(37769:48598)-S1Top(12861)-(S1Topc(37769:48598)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(37769:48598)+0.00019))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(12861:48598)+0.00019))),'r:','linewidth',2);hold on

plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1WellAxial(12861:14572)-S1WellAxial(12861)-(S1WellAxialc(12861:14572)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:14572)-1.2e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:48598)-1.2e-5))),'gr-','linewidth',1);hold on
plot(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726),(S1WellAxial(14572:22108)-S1WellAxial(12861)-(S1WellAxialc(14572:22108)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(14572:22108)-1.2e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:48598)-1.2e-5))),'gr-','linewidth',1);hold on
plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1WellAxial(22108:37769)-S1WellAxial(12861)-(S1WellAxialc(22108:37769)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(22108:37769)-1.2e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:48598)-1.2e-5))),'gr-','linewidth',1);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxial(37769:48598)-S1WellAxial(12861)-(S1WellAxialc(37769:48598)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(37769:48598)-1.2e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:48598)-1.2e-5))),'gr-','linewidth',1);hold on

xlim([500 4000]);
ylim([-1.1 1.1]);
grid on;
xlabel('Chamber Pressure psi');
ylabel('Normalized Displacements');
% title ('Pressure psi');

figure(76)

% subplot(1,2,1)
yyaxis left
plot(TimeD(12861:14572),(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726)),'k-','linewidth',3);hold on
plot(TimeD(14572:22108),(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726)),'c-','linewidth',3);hold on
plot(TimeD(22108:37769),(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726)),'r-','linewidth',3);hold on
plot(TimeD(37769:48598),(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726)),'b-','linewidth',3);hold on

yyaxis right
% plot(TimeD(12861:48598),S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)),'b-','linewidth',2);hold on
% plot(TimeD(12861:48598),S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)),'r-','linewidth',2);hold on
% plot(TimeD(12861:48598),4*(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))),'gr-','linewidth',2);hold on

plot(TimeD(12861:48598),S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(12861:48598)+7.8e-5),'b-','linewidth',2);hold on
plot(TimeD(12861:48598),S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(12861:48598)+0.00019),'r-','linewidth',2);hold on
plot(TimeD(12861:48598),4*(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861)))-4*(1.2e-8*Pz1RCfRCf(12861:48598)-1.2e-5),'gr-','linewidth',2);hold on
NumTicks = 20;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.5003896169 737202.518]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
% legend('Pressure (psi)','X-Yates','Y-Top','Z-Well Axial');
% legend('Chamber','Top Packer','Bottom Packer');
% title ('Pressure psi');

% stop

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%% filtrage de la partie hydromecanique lors de l ouverture de la
% %%%% fracture existante
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% figure(77)
% 
%  plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1Yates(12861:14572)-S1Yates(12861)-(S1Yatesc(12861:14572)-S1Yatesc(12861))),'k-','linewidth',3);hold on
% plot(Pz1RCfRCf(17219:18623)-Pz1RCfRCf(726),(S1Yates(17219:18623)-S1Yates(12861)-(S1Yatesc(17219:18623)-S1Yatesc(12861))),'c-','linewidth',3);hold on
% % plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Yates(22108:37769)-S1Yates(12861)-(S1Yatesc(22108:37769)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'r-','linewidth',3);hold on
% % plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Yates(37769:48598)-S1Yates(12861)-(S1Yatesc(37769:48598)-S1Yatesc(12861)))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)))),'b-','linewidth',3);hold on
% 
%  plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1Top(12861:14572)-S1Top(12861)-(S1Topc(12861:14572)-S1Topc(12861))),'k:','linewidth',2);hold on
% plot(Pz1RCfRCf(17219:18623)-Pz1RCfRCf(726),(S1Top(17219:18623)-S1Top(12861)-(S1Topc(17219:18623)-S1Topc(12861))),'c:','linewidth',2);hold on
% % plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Top(22108:37769)-S1Top(12861)-(S1Topc(22108:37769)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'r:','linewidth',2);hold on
% % plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Top(37769:48598)-S1Top(12861)-(S1Topc(37769:48598)-S1Topc(12861)))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)))),'b:','linewidth',2);hold on
% 
%  plot(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726),(S1WellAxial(12861:14572)-S1WellAxial(12861)-(S1WellAxialc(12861:14572)-S1WellAxialc(12861))),'k-','linewidth',1);hold on
% plot(Pz1RCfRCf(17219:18623)-Pz1RCfRCf(726),(S1WellAxial(17219:18623)-S1WellAxial(12861)-(S1WellAxialc(17219:18623)-S1WellAxialc(12861))),'c-','linewidth',1);hold on
% % plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1WellAxial(22108:37769)-S1WellAxial(12861)-(S1WellAxialc(22108:37769)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861)))),'r-','linewidth',1);hold on
% % plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxial(37769:48598)-S1WellAxial(12861)-(S1WellAxialc(37769:48598)-S1WellAxialc(12861)))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-(S1WellAxialc(12861))))),'b-','linewidth',1);hold on
% 
% xlim([500 4000]);
% % ylim([-1.1 1.1]);
% grid on;
% xlabel('Chamber Pressure psi');
% ylabel('Normalized Displacements');
% % title ('Pressure psi')
% 
% stop
% 
% figure(78)
% 
% plot(Pz1RCfRCf(12861:17219)-Pz1RCfRCf(726),(S1Yates(12861:17219)-S1Yates(12861)-(S1Yatesc(12861:17219)-S1Yatesc(12861))-(2.1e-7*Pz1RCfRCf(12861:17219)-0.0002))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(2.1e-7*Pz1RCfRCf(12861:48598)-0.0002))),'b-','linewidth',3);hold on
% plot(Pz1RCfRCf(17219:22108)-Pz1RCfRCf(726),(S1Yates(17219:22108)-S1Yates(12861)-(S1Yatesc(17219:22108)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(17219:22108)-0.00022))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(12861:48598)-0.00022))),'b-','linewidth',3);hold on
% plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Yates(22108:37769)-S1Yates(12861)-(S1Yatesc(22108:37769)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(22108:37769)-0.00022))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(12861:48598)-0.00022))),'b-','linewidth',3);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Yates(37769:48598)-S1Yates(12861)-(S1Yatesc(37769:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(37769:48598)-0.00022))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(12861:48598)-0.00022))),'b-','linewidth',3);hold on
% 
% plot(Pz1RCfRCf(12861:17219)-Pz1RCfRCf(726),(S1Top(12861:17219)-S1Top(12861)-(S1Topc(12861:17219)-S1Topc(12861))-(-3.6e-8*Pz1RCfRCf(12861:17219)+5e-5))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-3.6e-8*Pz1RCfRCf(12861:48598)+5e-5))),'r:','linewidth',2);hold on
% plot(Pz1RCfRCf(17219:22108)-Pz1RCfRCf(726),(S1Top(17219:22108)-S1Top(12861)-(S1Topc(17219:22108)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(17219:22108)+6.8e-5))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(12861:48598)+6.8e-5))),'r:','linewidth',2);hold on
% plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Top(22108:37769)-S1Top(12861)-(S1Topc(22108:37769)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(22108:37769)+6.8e-5))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(12861:48598)+6.8e-5))),'r:','linewidth',2);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Top(37769:48598)-S1Top(12861)-(S1Topc(37769:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(37769:48598)+6.8e-5))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(12861:48598)+6.8e-5))),'r:','linewidth',2);hold on
% 
% plot(Pz1RCfRCf(12861:17219)-Pz1RCfRCf(726),(S1WellAxial(12861:17219)-S1WellAxial(12861)-(S1WellAxialc(12861:17219)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:17219)-1.2e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:48598)-1.2e-5))),'gr-','linewidth',1);hold on
% plot(Pz1RCfRCf(17219:22108)-Pz1RCfRCf(726),(S1WellAxial(17219:22108)-S1WellAxial(12861)-(S1WellAxialc(17219:22108)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(17219:22108)-2.4e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(12861:48598)-2.4e-5))),'gr-','linewidth',1);hold on
% plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1WellAxial(22108:37769)-S1WellAxial(12861)-(S1WellAxialc(22108:37769)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(22108:37769)-2.4e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(12861:48598)-2.4e-5))),'gr-','linewidth',1);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxial(37769:48598)-S1WellAxial(12861)-(S1WellAxialc(37769:48598)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(37769:48598)-2.4e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(12861:48598)-2.4e-5))),'gr-','linewidth',1);hold on
% 
% xlim([500 4500]);
% ylim([-1.1 1.1]);
% grid on;
% xlabel('Chamber Pressure psi');
% ylabel('Normalized Displacements');
% % title ('Pressure psi');
% 
% figure(780)
% 
% % plot(Pz1RCfRCf(12861:17219)-Pz1RCfRCf(726),(S1Yates(12861:17219)-S1Yates(12861)-(S1Yatesc(12861:17219)-S1Yatesc(12861))-(2.1e-7*Pz1RCfRCf(12861:17219)-0.0002))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(2.1e-7*Pz1RCfRCf(12861:48598)-0.0002))),'b-','linewidth',3);hold on
% % plot(Pz1RCfRCf(17219:22108)-Pz1RCfRCf(726),(S1Yates(17219:22108)-S1Yates(12861)-(S1Yatesc(17219:22108)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(17219:22108)-0.00022))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(12861:48598)-0.00022))),'b-','linewidth',3);hold on
% % plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Yates(22108:37769)-S1Yates(12861)-(S1Yatesc(22108:37769)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(22108:37769)-0.00022))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(12861:48598)-0.00022))),'b-','linewidth',3);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Yates(37769:48598)-S1Yates(12861)-(S1Yatesc(37769:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(37769:48598)-0.00022))/max(abs(S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(12861:48598)-0.00022))),'b-','linewidth',3);hold on
% 
% % plot(Pz1RCfRCf(12861:17219)-Pz1RCfRCf(726),(S1Top(12861:17219)-S1Top(12861)-(S1Topc(12861:17219)-S1Topc(12861))-(-3.6e-8*Pz1RCfRCf(12861:17219)+5e-5))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-3.6e-8*Pz1RCfRCf(12861:48598)+5e-5))),'r:','linewidth',2);hold on
% % plot(Pz1RCfRCf(17219:22108)-Pz1RCfRCf(726),(S1Top(17219:22108)-S1Top(12861)-(S1Topc(17219:22108)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(17219:22108)+6.8e-5))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(12861:48598)+6.8e-5))),'r:','linewidth',2);hold on
% % plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1Top(22108:37769)-S1Top(12861)-(S1Topc(22108:37769)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(22108:37769)+6.8e-5))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(12861:48598)+6.8e-5))),'r:','linewidth',2);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1Top(37769:48598)-S1Top(12861)-(S1Topc(37769:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(37769:48598)+6.8e-5))/max(abs(S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(12861:48598)+6.8e-5))),'r:','linewidth',2);hold on
% 
% % plot(Pz1RCfRCf(12861:17219)-Pz1RCfRCf(726),(S1WellAxial(12861:17219)-S1WellAxial(12861)-(S1WellAxialc(12861:17219)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:17219)-1.2e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.2e-8*Pz1RCfRCf(12861:48598)-1.2e-5))),'gr-','linewidth',1);hold on
% % plot(Pz1RCfRCf(17219:22108)-Pz1RCfRCf(726),(S1WellAxial(17219:22108)-S1WellAxial(12861)-(S1WellAxialc(17219:22108)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(17219:22108)-2.4e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(12861:48598)-2.4e-5))),'gr-','linewidth',1);hold on
% % plot(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726),(S1WellAxial(22108:37769)-S1WellAxial(12861)-(S1WellAxialc(22108:37769)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(22108:37769)-2.4e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(12861:48598)-2.4e-5))),'gr-','linewidth',1);hold on
% plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxial(37769:48598)-S1WellAxial(12861)-(S1WellAxialc(37769:48598)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(37769:48598)-2.4e-5))/max(abs(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))-(1.7e-8*Pz1RCfRCf(12861:48598)-2.4e-5))),'gr-','linewidth',1);hold on
% 
% xlim([500 4000]);
% ylim([-1.1 1.1]);
% grid on;
% xlabel('Chamber Pressure psi');
% ylabel('Normalized Displacements');
% % title ('Pressure psi');
% % 
% figure(79)
% 
% % subplot(1,2,1)
% yyaxis left
%  
% % plot(TimeD(17219:18623),(Pz1RCfRCf(17219:18623)-Pz1RCfRCf(726)),'k-','linewidth',5);hold on
% plot(TimeD(12861:14572),(Pz1RCfRCf(12861:14572)-Pz1RCfRCf(726)),'k-','linewidth',3);hold on
% plot(TimeD(14572:22108),(Pz1RCfRCf(14572:22108)-Pz1RCfRCf(726)),'c-','linewidth',3);hold on
% plot(TimeD(22108:37769),(Pz1RCfRCf(22108:37769)-Pz1RCfRCf(726)),'r-','linewidth',3);hold on
% plot(TimeD(37769:48598),(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726)),'b-','linewidth',3);hold on
% 
% yyaxis right
% % plot(TimeD(12861:48598),S1Yates(12861:48598)-S1Yates(12861)-(S1Yatesc(12861:48598)-S1Yatesc(12861)),'b-','linewidth',2);hold on
% % plot(TimeD(12861:48598),S1Top(12861:48598)-S1Top(12861)-(S1Topc(12861:48598)-S1Topc(12861)),'r-','linewidth',2);hold on
% % plot(TimeD(12861:48598),4*(S1WellAxial(12861:48598)-S1WellAxial(12861)-(S1WellAxialc(12861:48598)-S1WellAxialc(12861))),'gr-','linewidth',2);hold on
% plot(TimeD(12861:17219),S1Yates(12861:17219)-S1Yates(12861)-(S1Yatesc(12861:17219)-S1Yatesc(12861))-(2.1e-7*Pz1RCfRCf(12861:17219)-0.0002),'b-','linewidth',2);hold on
% plot(TimeD(17219:48598),S1Yates(17219:48598)-S1Yates(12861)-(S1Yatesc(17219:48598)-S1Yatesc(12861))-(2.2e-7*Pz1RCfRCf(17219:48598)-0.00022),'b-','linewidth',2);hold on
% 
% plot(TimeD(12861:17219),S1Top(12861:17219)-S1Top(12861)-(S1Topc(12861:17219)-S1Topc(12861))-(-3.6e-8*Pz1RCfRCf(12861:17219)+5e-5),'r-','linewidth',2);hold on
% plot(TimeD(17219:48598),S1Top(17219:48598)-S1Top(12861)-(S1Topc(17219:48598)-S1Topc(12861))-(-4.3e-8*Pz1RCfRCf(17219:48598)+6.8e-5),'r-','linewidth',2);hold on
% 
% plot(TimeD(12861:17219),4*(S1WellAxial(12861:17219)-S1WellAxial(12861)-(S1WellAxialc(12861:17219)-S1WellAxialc(12861)))-4*(1.2e-8*Pz1RCfRCf(12861:17219)-1.2e-5),'gr-','linewidth',2);hold on
% plot(TimeD(17219:48598),4*(S1WellAxial(17219:48598)-S1WellAxial(12861)-(S1WellAxialc(17219:48598)-S1WellAxialc(12861)))-4*(1.7e-8*Pz1RCfRCf(17219:48598)-2.4e-5),'gr-','linewidth',2);hold on
% 
% NumTicks = 20;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','HH:MM','keeplimits','keepticks');
% % xlim([737202.5003896169 737202.518]);
% %ylim([-1 40]);
% grid on;
% %xlabel('Date');
% % legend('Pressure (psi)','X-Yates','Y-Top','Z-Well Axial');
% % legend('Chamber','Top Packer','Bottom Packer');
% % title ('Pressure psi');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filtered Elastic and Inelastic Displacements

S1YatesE = cat(1,Pz1RCf(1:12861)-Pz1RCf(1:12861),(-6.5e-8*Pz1RCfRCf(12861:82721)+7.8e-5));
S1TopE = cat(1,Pz1RCf(1:12861)-Pz1RCf(1:12861),(-2e-7*Pz1RCfRCf(12861:82721)+0.00019));
S1WellAxialE = cat(1,4*(Pz1RCf(1:12861)-Pz1RCf(1:12861)),4*(1.2e-8*Pz1RCfRCf(12861:82721)-1.2e-5));

S1YatesIE = cat(1,S1Yates(1:12861)-S1Yates(1:12861),S1Yates(12861:82721)-S1Yates(12861)-(S1Yatesc(12861:82721)-S1Yatesc(12861))-(-6.5e-8*Pz1RCfRCf(12861:82721)+7.8e-5));
S1TopIE = cat(1,S1Top(1:12861)-S1Top(1:12861),S1Top(12861:82721)-S1Top(12861)-(S1Topc(12861:82721)-S1Topc(12861))-(-2e-7*Pz1RCfRCf(12861:82721)+0.00019));
S1WellAxialIE = cat(1,4*(S1WellAxial(1:12861)-S1WellAxial(1:12861)),4*(S1WellAxial(12861:82721)-S1WellAxial(12861)-(S1WellAxialc(12861:82721)-S1WellAxialc(12861)))-4*(1.2e-8*Pz1RCfRCf(12861:82721)-1.2e-5));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Enregistrement en .mat et en .txt of reversible and irreversible components

% SIGMAV_SIMFIPSURF05232018 = [Pz1RCf(1:82721),S1Yates(1:82721),S1Top(1:82721),S1WellAxial(1:82721),S1YatesE(1:82721),S1TopE(1:82721),S1WellAxialE(1:82721),S1YatesIE(1:82721),S1TopIE(1:82721),S1WellAxialIE(1:82721)];
% save('SIGMAV_SIMFIPSURF05232018.txt','SIGMAV_SIMFIPSURF05232018', '-ASCII');
% save ('SIGMAV_SIMFIPSURF05232018.mat');
%
date=datestr(TimeD(1:82721))
header='date Pz1 S1Yates S1Top S1WellAxial S1YatesE S1TopE S1WellAxialE S1YatesIE S1TopIE S1WellAxialIE '
M=[cellstr(date) num2cell(Pz1RCf(1:82721))  num2cell(S1Yates(1:82721))  num2cell(S1Top(1:82721))  num2cell(S1WellAxial(1:82721))  num2cell(S1YatesE(1:82721))  num2cell(S1TopE(1:82721))  num2cell(S1WellAxialE(1:82721))  num2cell(S1YatesIE(1:82721))  num2cell(S1TopIE(1:82721))  num2cell(S1WellAxialIE(1:82721))]
fid = fopen('SIGMAV_SIMFIPSURF05232018.txt','w');
format long;
% myformat='%s %12.4f %12.4f \r\n'
myformat='%s %f %1.15f %1.15f %1.15f %1.15f %1.15f %1.15f %1.15f %1.15f %1.15f \r\n'
fprintf(fid,'%s\r\n',header);
for k=1:size(M,1)
   fprintf(fid,myformat,M{k,:})
end
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure (8)

subplot(2,1,1)
yyaxis left
plot(TimeD(1:82721),(Pz1RCfRCf(1:82721)-Pz1RCfRCf(1)),'c-','linewidth',3);hold on
ylabel('Fluid Pressure (psi)');

yyaxis right
plot(TimeD(1:82721),S1YatesE(1:82721),'b-','linewidth',2);hold on
% plot(TimeD(1:48598),S1YatesIE(1:48598),'b:','linewidth',2);hold on
% 
% plot(TimeD(12861:48598),S1TopE(12861:48598),'r-','linewidth',2);hold on
% % plot(TimeD(1:48598),S1TopIE(1:48598),'r:','linewidth',2);hold on
% 
% plot(TimeD(12861:48598),S1WellAxialE(12861:48598),'gr-','linewidth',2);hold on
% % plot(TimeD(1:48598),S1WellAxialIE(1:48598),'gr:','linewidth',2);hold on
ylabel('Displacement (micron)');
grid on;
NumTicks = 25;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.66 737202.675]);

subplot(2,1,2)
yyaxis left
plot(TimeD(12861:48598),(Pz1RCfRCf(12861:48598)-Pz1RCfRCf(726)),'c-','linewidth',3);hold on
ylabel('Fluid Pressure (psi)');

yyaxis right
% plot(TimeD(1:48598),S1YatesE(1:48598),'b:','linewidth',2);hold on
plot(TimeD(12861:48598),S1YatesIE(12861:48598),'b-','linewidth',2);hold on

% plot(TimeD(1:48598),S1TopE(1:48598),'r:','linewidth',2);hold on
plot(TimeD(12861:48598),S1TopIE(12861:48598),'r-','linewidth',2);hold on

% plot(TimeD(1:48598),S1WellAxialE(1:48598),'gr:','linewidth',2);hold on
plot(TimeD(12861:48598),S1WellAxialIE(12861:48598),'gr-','linewidth',2);hold on
ylabel('Displacement (micron)');
NumTicks = 25;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.66 737202.675]);
grid on;

figure(80)


plot(Pz1RCfRCf(12861:37769)-Pz1RCfRCf(726),sqrt((S1YatesIE(12861:37769)-S1YatesIE(12861)).^2+(S1TopIE(12861:37769)-S1TopIE(12861)).^2),'c:','linewidth',3);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),sqrt((S1YatesIE(37769:48598)-S1YatesIE(12861)).^2+(S1TopIE(37769:48598)-S1TopIE(12861)).^2),'r:','linewidth',3);hold on

plot(Pz1RCfRCf(12861:37769)-Pz1RCfRCf(726),(S1WellAxialIE(12861:37769)-S1WellAxialIE(12861)),'c-','linewidth',1);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxialIE(37769:48598)-S1WellAxialIE(12861)),'r-','linewidth',1);hold on

xlim([500 4000]);
% ylim([-1.1 1.1]);
grid on;
xlabel('Chamber Pressure psi');
ylabel('Displacements');
legend('Radial',' ','Axial',' ');
% title ('Pressure psi')

figure (81)

subplot(2,1,1)

plot3(S1YatesE(12861:48598),S1WellAxialE(12861:48598)*-1,S1TopE(12861:48598),'b-','linewidth',2);hold on
xlabel('S1EEast');
ylabel('S1ENord');
zlabel('S1EUp');
xlim([-2e-4 8e-4]);
ylim([-2e-4 8e-4]);
zlim([-4e-4 6e-4]);
axis square;
grid on;
title('Elastic not rotated');
view(44,20);
subplot(2,1,2)

plot3(S1YatesIE(12861:48598),S1WellAxialIE(12861:48598)*-1,S1TopIE(12861:48598),'b-','linewidth',2);hold on
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
xlim([-2e-4 8e-4]);
ylim([-2e-4 8e-4]);
zlim([-4e-4 6e-4]);
axis square;
grid on;
title('InElastic not rotated');
view(44,20);


%  stop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stereoplots
%
% Sonde Stimulation - Rotation des coordonnees de Oz axe forage, Oy
% vertical, Ox vers Yates vers Oz vertical (S1Up), Oy Nord (S1Nord) et Ox Est (S1East)

angS1XE = pi*(0/180); % The angle (0 < angX < 360� )is around Ox  
angS1YE = pi*(-12/180); % The angle (0 < angY < 360� )is around Oy  
angS1ZE = pi*(0/180); % The angle (0 < angZ < 360� )is around Oz  
% angS1Z = 50 ; % The angle (0 < angZ < 360� )is around Oz
%  rS1xE = [1,0,0;0,cos(angS1XE),-sin(angS1XE);0,sin(angS1XE),cos(angS1XE)];
rS1yE = [cos(angS1YE),0,sin(angS1YE);0,1,0;-sin(angS1YE),0,cos(angS1YE)];% 
% rS1z = [cos(angS1Z),-sin(angS1Z),0;sin(angS1Z),cos(angS1Z),0;0,0,1];% 

calculS1E = [];
S1EEast = [];
S1ENord = [];
S1EUp = [];

for ni = 12861:48598
calculS1E =  [S1WellAxialE(ni,1);S1YatesE(ni,1);S1TopE(ni,1)];  
rotatS1E = rS1yE*calculS1E;
S1ENord(ni,1) = rotatS1E(1,1)*-1;
S1EEast(ni,1) = rotatS1E(2,1);
S1EUp(ni,1) = rotatS1E(3,1);
end

calculS1IE = [];
S1IEEast = [];
S1IENord = [];
S1IEUp = [];

for ni = 12861:48598
calculS1IE =  [S1WellAxialIE(ni,1);S1YatesIE(ni,1);S1TopIE(ni,1)];  
rotatS1IE = rS1yE*calculS1IE;
S1IENord(ni,1) = rotatS1IE(1,1)*-1;
S1IEEast(ni,1) = rotatS1IE(2,1);
S1IEUp(ni,1) = rotatS1IE(3,1);
end

for ni = 12861:48598
calculS1Tot =  [S1WellAxial(ni,1)-(S1WellAxialc(ni,1));S1Yates(ni,1)-S1Yatesc(ni,1);S1Top(ni,1)-S1Topc(ni,1)];  
rotatS1Tot = rS1yE*calculS1Tot;
S1TotNord(ni,1) = rotatS1Tot(2,1)*-1;
S1TotEast(ni,1) = rotatS1Tot(1,1);
S1TotUp(ni,1) = rotatS1Tot(3,1);
end

% figure(25)
% % Verification des orientations apres changement de repere
% subplot(1,3,1)
% plot(S1YatesIE(5589:25432),S1WellAxialIE(5589:25432),'b:','linewidth',2);hold on
% plot(S1IEEast(5589:25432),S1IENord(5589:25432),'b-','linewidth',2);hold on
% xlabel('S1YatesIE-S1IEEast');
% ylabel('S1WellAxialIE-S1IENord');
% axis square;
% 
% subplot(1,3,2)
% plot(S1YatesIE(5589:25432),S1TopIE(5589:25432),'b:','linewidth',2);hold on
% plot(S1IEEast(5589:25432),S1IEUp(5589:25432),'b-','linewidth',2);hold on
% xlabel('S1YatesIE-S1IEEast');
% ylabel('S1TopIE-S1IEUp');
% axis square;
% 
% subplot(1,3,3)
% plot(S1WellAxialIE(5589:25432),S1TopIE(5589:25432),'b:','linewidth',2);hold on
% plot(S1IENord(5589:25432),S1IEUp(5589:25432),'b-','linewidth',2);hold on
% xlabel('S1WellAxialIE-S1IENord');
% ylabel('S1TopIE-S1IEUp');
% axis square;

%

%Borehole coordinates
xbore = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
ybore = [-4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12]'*1e-4
zbore = [0.85 0.64 0.43 0.21 0 -0.21 -0.43 -0.64 -0.85 -1.06 -1.28 -1.49 -1.7 -1.91 -2.13 -2.34 -2.55]'*1e-4

% Verification des orientations apres changement de repere
figure(9)
% plot(S1YatesIE(12861:37769)-S1YatesIE(12861),S1WellAxialIE(12861:37769)-S1WellAxialIE(12861),'c-','linewidth',2);hold on
% plot(S1YatesIE(37769:48598)-S1YatesIE(12861),S1WellAxialIE(37769:48598)-S1WellAxialIE(12861),'r-','linewidth',2);hold on
% xlim([-3e-4 1e-4]);
% ylim([-1e-4 3e-4]);

plot(S1IEEast(12861:37769)-S1IEEast(12861),S1IENord(12861:37769)-S1IENord(12861),'c-','linewidth',2);hold on
plot(S1IEEast(37769:48598)-S1IEEast(12861),S1IENord(37769:48598)-S1IENord(12861),'r-','linewidth',2);hold on
xlim([-3e-4 1e-4]);
ylim([-3e-4 1e-4]);

xlabel('S1YatesIE-S1IEEast');
ylabel('S1WellAxialIE-S1IENord');
axis square;

figure(91)
% plot(S1WellAxialIE(12861:37769)-S1WellAxialIE(12861),S1TopIE(12861:37769)-S1TopIE(12861),'c-','linewidth',2);hold on
% plot(S1WellAxialIE(37769:48598)-S1WellAxialIE(12861),S1TopIE(37769:48598)-S1TopIE(12861),'r-','linewidth',2);hold on
% xlim([-1e-4 3e-4]);
% ylim([-3.5e-4 0.5e-4]);

plot(S1IENord(12861:37769)-S1IENord(12861),S1IEUp(12861:37769)-S1IEUp(12861),'c-','linewidth',2);hold on
plot(S1IENord(37769:48598)-S1IENord(12861),S1IEUp(37769:48598)-S1IEUp(12861),'r-','linewidth',2);hold on
xlim([-2.5e-4 1.5e-4]);
ylim([-3.5e-4 0.5e-4]);
xlabel('S1WellAxialIE-S1IENord');
ylabel('S1TopIE-S1IEUp');
axis square;
% 
figure (92)

plot3(S1IEEast(12861:37769)-S1IEEast(12861),S1IENord(12861:37769)-S1IENord(12861),S1IEUp(12861:37769)-S1IEUp(12861),'c-','linewidth',2);hold on
plot3(S1IEEast(37769:48598)-S1IEEast(12861),S1IENord(37769:48598)-S1IENord(12861),S1IEUp(37769:48598)-S1IEUp(12861),'r-','linewidth',2);hold on
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
axis square;
grid on;
xlim([-3e-4 2e-4]);
ylim([-4e-4 1e-4]);
zlim([-4e-4 1e-4]);
view(44,20);
axis square;
title('Upper Anchor - NPD');

figure (93)

plot3(S1IEEast(12861:37769)-S1IEEast(12861),S1IENord(12861:37769)-S1IENord(12861),S1IEUp(12861:37769)-S1IEUp(12861),'c-','linewidth',2);hold on
plot3(S1IEEast(37769:48598)-S1IEEast(12861),S1IENord(37769:48598)-S1IENord(12861),S1IEUp(37769:48598)-S1IEUp(12861),'r-','linewidth',2);hold on
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
axis square;
grid on;
xlim([-3e-4 2e-4]);
ylim([-4e-4 1e-4]);
zlim([-4e-4 1e-4]);
view(90,0);
axis square;
title('Upper Anchor - NPD');

figure (94)

plot3(S1IEEast(12861:37769)-S1IEEast(12861),S1IENord(12861:37769)-S1IENord(12861),S1IEUp(12861:37769)-S1IEUp(12861),'c-','linewidth',2);hold on
plot3(S1IEEast(37769:48598)-S1IEEast(12861),S1IENord(37769:48598)-S1IENord(12861),S1IEUp(37769:48598)-S1IEUp(12861),'r-','linewidth',2);hold on
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
axis square;
grid on;
xlim([-3e-4 2e-4]);
ylim([-4e-4 1e-4]);
zlim([-4e-4 1e-4]);
view(0,0);
axis square;
title('Upper Anchor - NPD');

figure (95)

plot3(S1IEEast(12861:37769)-S1IEEast(12861),S1IENord(12861:37769)-S1IENord(12861),S1IEUp(12861:37769)-S1IEUp(12861),'c-','linewidth',2);hold on
plot3(S1IEEast(37769:48598)-S1IEEast(12861),S1IENord(37769:48598)-S1IENord(12861),S1IEUp(37769:48598)-S1IEUp(12861),'r-','linewidth',2);hold on
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
axis square;
grid on;
xlim([-3e-4 2e-4]);
ylim([-4e-4 1e-4]);
zlim([-4e-4 1e-4]);
view(0,90);
axis square;
title('Upper Anchor - NPD');

figure (96)

plot(TimeD(12861:37769),(Pz1RCfRCf(12861:37769)-Pz1RCfRCf(726)),'c-','linewidth',2);hold on
plot(TimeD(37769:48598),(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726)),'r-','linewidth',2);hold on
ylabel('Fluid Pressure (psi)');
NumTicks = 25;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.66 737202.675]);
%ylim([-1 40]);
grid on;
xlabel('Local Time (hh:mm)');
ylabel('Pressure (psi)');

% stop

% % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % %%%%  Calcul par changement de repere d ouverture et slip sur fracture
% % % % % %%%%  potentiellement activee
% % % % % 
% % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % Stereoplots
% % % % % %
% % % % % % Sonde Stimulation - Rotation des coordonnees de Oz axe forage, Oy
% % % % % % vertical, Ox vers Yates vers Oz vertical (S1Up), Oy Nord (S1Nord) et Ox Est (S1East)
% % % % % 
% % % % % % Case N114-49SW
% % % % % angS1XHF = pi*(-49/180); % The angle (0 < angX < 360� )is around Ox  
% % % % % angS1YHF = pi*(0/180); % The angle (0 < angY < 360� )is around Oy  
% % % % % angS1ZHF = pi*(24/180); % The angle (0 < angZ < 360� )is around Oz  
% % % % % % angS1Z = 50 ; % The angle (0 < angZ < 360� )is around Oz
% % % % %  rS1xHF = [1,0,0;0,cos(angS1XHF),-sin(angS1XHF);0,sin(angS1XHF),cos(angS1XHF)];
% % % % %  rS1yHF = [cos(angS1YHF),0,sin(angS1YHF);0,1,0;-sin(angS1YHF),0,cos(angS1YHF)];% 
% % % % %  rS1zHF = [cos(angS1ZHF),-sin(angS1ZHF),0;sin(angS1ZHF),cos(angS1ZHF),0;0,0,1];% 
% % % % % 
% % % % % calculS1HF = [];
% % % % % S1HFHorizontalSlip = [];
% % % % % S1HFUpSlip = [];
% % % % % S1HFNormal = [];
% % % % % 
% % % % % % Northern10Hz = S1IENord(12861:48598);
% % % % % % Western10Hz = S1IEEast(12861:48598); %Attention
% % % % % % Vertical10Hz = S1IEUp(12861:48598);
% % % % % 
% % % % % for ni = 1:48598
% % % % % calculS1HF =  [S1IEEast(ni,1);S1IENord(ni,1)*1;S1IEUp(ni,1)];  
% % % % %  rotatS1HF = rS1zHF*rS1xHF*calculS1HF;
% % % % % %rotatS1HF = rS1xHF*calculS1HF;
% % % % % S1HFHorizontalSlip(ni,1) = rotatS1HF(1,1);
% % % % % S1HFUpSlip(ni,1) = rotatS1HF(2,1);
% % % % % S1HFNormal(ni,1) = rotatS1HF(3,1);
% % % % % end
% % % % % 
% % % % % % Case N124-48SW
% % % % % angS1XHF1 = pi*(-48/180); % The angle (0 < angX < 360� )is around Ox  
% % % % % angS1YHF1 = pi*(0/180); % The angle (0 < angY < 360� )is around Oy  
% % % % % angS1ZHF1 = pi*(34/180); % The angle (0 < angZ < 360� )is around Oz  
% % % % % % angS1Z = 50 ; % The angle (0 < angZ < 360� )is around Oz
% % % % %  rS1xHF1 = [1,0,0;0,cos(angS1XHF1),-sin(angS1XHF1);0,sin(angS1XHF1),cos(angS1XHF1)];
% % % % %  rS1yHF1 = [cos(angS1YHF1),0,sin(angS1YHF1);0,1,0;-sin(angS1YHF1),0,cos(angS1YHF1)];% 
% % % % %  rS1zHF1 = [cos(angS1ZHF1),-sin(angS1ZHF1),0;sin(angS1ZHF1),cos(angS1ZHF1),0;0,0,1];% 
% % % % % 
% % % % % calculS1HF1 = [];
% % % % % S1HFHorizontalSlip1 = [];
% % % % % S1HFUpSlip1 = [];
% % % % % S1HFNormal1= [];
% % % % % 
% % % % % for ni = 1:48598
% % % % % calculS1HF1 =  [S1IEEast(ni,1);S1IENord(ni,1)*1;S1IEUp(ni,1)];  
% % % % %  rotatS1HF1 = rS1zHF1*rS1xHF1*calculS1HF1;
% % % % % %rotatS1HF = rS1xHF*calculS1HF;
% % % % % S1HFHorizontalSlip1(ni,1) = rotatS1HF1(1,1);
% % % % % S1HFUpSlip1(ni,1) = rotatS1HF1(2,1);
% % % % % S1HFNormal1(ni,1) = rotatS1HF1(3,1);
% % % % % end
% % % % % 
% % % % % % Case N83-75SW
% % % % % angS1XHF2 = pi*(-75/180); % The angle (0 < angX < 360� )is around Ox  
% % % % % angS1YHF2 = pi*(0/180); % The angle (0 < angY < 360� )is around Oy  
% % % % % angS1ZHF2 = pi*(-7/180); % The angle (0 < angZ < 360� )is around Oz  
% % % % % % angS1Z = 50 ; % The angle (0 < angZ < 360� )is around Oz
% % % % %  rS1xHF2 = [1,0,0;0,cos(angS1XHF2),-sin(angS1XHF2);0,sin(angS1XHF2),cos(angS1XHF2)];
% % % % %  rS1yHF2 = [cos(angS1YHF2),0,sin(angS1YHF2);0,1,0;-sin(angS1YHF2),0,cos(angS1YHF2)];% 
% % % % %  rS1zHF2 = [cos(angS1ZHF2),-sin(angS1ZHF2),0;sin(angS1ZHF2),cos(angS1ZHF2),0;0,0,1];% 
% % % % % 
% % % % % calculS1HF2 = [];
% % % % % S1HFHorizontalSlip2 = [];
% % % % % S1HFUpSlip2 = [];
% % % % % S1HFNormal2= [];
% % % % % 
% % % % % for ni = 1:48598
% % % % % calculS1HF2 =  [S1IEEast(ni,1);S1IENord(ni,1)*1;S1IEUp(ni,1)];  
% % % % %  rotatS1HF2 = rS1zHF2*rS1xHF2*calculS1HF2;
% % % % % %rotatS1HF = rS1xHF*calculS1HF;
% % % % % S1HFHorizontalSlip2(ni,1) = rotatS1HF2(1,1);
% % % % % S1HFUpSlip2(ni,1) = rotatS1HF2(2,1);
% % % % % S1HFNormal2(ni,1) = rotatS1HF2(3,1);
% % % % % end
% % % % % 
% % % % % figure(960000)
% % % % % subplot(3,1,1)
% % % % % plot(TimeD(12861:48598),(Pz1RCfRCf(12861:48598)-Pz1RCfRCf(726)),'c-','linewidth',3);hold on
% % % % % ylabel('Fluid Pressure (psi)');
% % % % % xlabel('time (seconds)');
% % % % % grid on;
% % % % % NumTicks = 25;
% % % % % L = get(gca,'XLim');
% % % % % set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % % % % datetick('x','HH:MM','keeplimits','keepticks');
% % % % % 
% % % % % subplot(3,1,2)
% % % % % plot(TimeD(12861:48598),S1HFHorizontalSlip(12861:48598)-S1HFHorizontalSlip(12861),'b-','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),S1HFUpSlip(12861:48598)-S1HFUpSlip(12861),'r-','linewidth',2);hold on
% % % % % 
% % % % % plot(TimeD(12861:48598),S1HFHorizontalSlip1(12861:48598)-S1HFHorizontalSlip1(12861),'b--','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),S1HFUpSlip1(12861:48598)-S1HFUpSlip1(12861),'r--','linewidth',2);hold on
% % % % % 
% % % % % plot(TimeD(12861:48598),S1HFHorizontalSlip2(12861:48598)-S1HFHorizontalSlip2(12861),'b:','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),S1HFUpSlip2(12861:48598)-S1HFUpSlip2(12861),'r:','linewidth',2);hold on
% % % % % 
% % % % % xlabel('time (seconds)');
% % % % % ylabel('Displacement (micron)');
% % % % % NumTicks = 25;
% % % % % L = get(gca,'XLim');
% % % % % set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % % % % datetick('x','HH:MM','keeplimits','keepticks');
% % % % % legend('N114-49SW Horizontal shear','N114-49SW Up shear','N124-48SW Horizontal shear','N124-48SW Up shear','N86-69SW Horizontal shear','N86-69SW Up shear');
% % % % % grid on;
% % % % % 
% % % % % subplot(3,1,3)
% % % % % plot(TimeD(12861:48598),S1HFNormal(12861:48598)-S1HFNormal(12861),'gr-','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),S1HFNormal1(12861:48598)-S1HFNormal1(12861),'gr--','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),S1HFNormal2(12861:48598)-S1HFNormal2(12861),'gr:','linewidth',2);hold on
% % % % % xlabel('time (seconds)');
% % % % % ylabel('Displacement (micron)');
% % % % % NumTicks = 25;
% % % % % L = get(gca,'XLim');
% % % % % set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % % % % datetick('x','HH:MM','keeplimits','keepticks');
% % % % % legend('N114-49SW Normal opening','N124-48SW Normal opening','N86-69SW Normal opening');
% % % % % grid on;
% % % % % 
% % % % % figure(960001)
% % % % % 
% % % % % subplot(2,1,1)
% % % % % plot(TimeD(12861:48598),(Pz1RCfRCf(12861:48598)-Pz1RCfRCf(726)),'c-','linewidth',3);hold on
% % % % % ylabel('Fluid Pressure (psi)');
% % % % % xlabel('time (seconds)');
% % % % % grid on;
% % % % % NumTicks = 25;
% % % % % L = get(gca,'XLim');
% % % % % set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % % % % datetick('x','HH:MM','keeplimits','keepticks');
% % % % % 
% % % % % subplot(2,1,2)
% % % % % 
% % % % % % tots=sqrt((S1HFHorizontalSlip(12861:48598)-S1HFHorizontalSlip(12861)).^2+(S1HFUpSlip(12861:48598)-S1HFUpSlip(12861))).^2);
% % % % % plot(TimeD(12861:48598),sqrt((S1HFHorizontalSlip(12861:48598)-S1HFHorizontalSlip(12861)).^2+(S1HFUpSlip(12861:48598)-S1HFUpSlip(12861)).^2),'b-','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),sqrt((S1HFHorizontalSlip1(12861:48598)-S1HFHorizontalSlip1(12861)).^2+(S1HFUpSlip1(12861:48598)-S1HFUpSlip1(12861)).^2),'b--','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),sqrt((S1HFHorizontalSlip2(12861:48598)-S1HFHorizontalSlip2(12861)).^2+(S1HFUpSlip2(12861:48598)-S1HFUpSlip2(12861)).^2),'b:','linewidth',2);hold on
% % % % % 
% % % % % plot(TimeD(12861:48598),S1HFNormal(12861:48598)-S1HFNormal(12861),'gr-','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),S1HFNormal1(12861:48598)-S1HFNormal1(12861),'gr--','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),S1HFNormal2(12861:48598)-S1HFNormal2(12861),'gr:','linewidth',2);hold on
% % % % % xlabel('time (seconds)');
% % % % % ylabel('Displacement (m)');
% % % % % NumTicks = 25;
% % % % % L = get(gca,'XLim');
% % % % % set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % % % % datetick('x','HH:MM','keeplimits','keepticks');
% % % % % legend('N114-49SW Shear','N124-48SW Shear','N86-69SW Shear','N114-49SW Normal opening','N124-48SW Normal opening','N86-69SW Normal opening');
% % % % % grid on;
% % % % % 
% % % % % % stop
% % % % % 
% % % % % figure(960)
% % % % % 
% % % % % % subplot(2,1,1)
% % % % % plot3(S1HFHorizontalSlip(12861:37769)-S1HFHorizontalSlip(12861),S1HFUpSlip(12861:37769)-S1HFUpSlip(12861),S1HFNormal(12861:37769)-S1HFNormal(12861),'gr-','linewidth',2);hold on
% % % % % plot3(S1HFHorizontalSlip(37769:48598)-S1HFHorizontalSlip(12861),S1HFUpSlip(37769:48598)-S1HFUpSlip(12861),S1HFNormal(37769:48598)-S1HFNormal(12861),'gr:','linewidth',2);hold on
% % % % % grid on;
% % % % % xlabel('Horizontal slip');
% % % % % ylabel('Up Slip');
% % % % % zlabel('Normal opening');
% % % % % % xlim([-0.5e-4 3e-4]);
% % % % % % ylim([-2.5e-4 1.5e-4]);
% % % % % % zlim([-3e-4 0.5e-4]);
% % % % % view(44,20);
% % % % % axis square
% % % % % 
% % % % % figure(961)
% % % % % 
% % % % % % subplot(2,1,2)
% % % % % plot3(S1IEEast(12861:37769),S1IENord(12861:37769),S1IEUp(12861:37769),'r-','linewidth',2);hold on
% % % % % plot3(S1IEEast(37769:48598),S1IENord(37769:48598),S1IEUp(37769:48598),'r:','linewidth',2);hold on
% % % % % plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
% % % % % grid on;
% % % % % xlabel('S1IEEast');
% % % % % ylabel('S1IENord');
% % % % % zlabel('S1IEUp');
% % % % % xlim([-0.5e-4 3e-4]);
% % % % % ylim([-2.5e-4 1.5e-4]);
% % % % % zlim([-3e-4 0.5e-4]);
% % % % % view(44,20);
% % % % % axis square
% % % % % 
% % % % % figure(96)
% % % % % 
% % % % % yyaxis left
% % % % % plot(TimeD(12861:48598),(Pz1RCfRCf(12861:48598)-Pz1RCfRCf(726)),'c-','linewidth',3);hold on
% % % % % ylabel('Fluid Pressure (psi)');
% % % % % 
% % % % % yyaxis right
% % % % % 
% % % % % plot(TimeD(12861:48598),S1HFHorizontalSlip(12861:48598)-S1HFHorizontalSlip(12861),'b-','linewidth',2);hold on
% % % % % plot(TimeD(12861:48598),S1HFUpSlip(12861:48598)-S1HFUpSlip(12861),'r-','linewidth',2);hold on
% % % % % 
% % % % % plot(TimeD(12861:48598),S1HFNormal(12861:48598)-S1HFNormal(12861),'gr-','linewidth',2);hold on
% % % % % xlabel('time (seconds)');
% % % % % ylabel('Displacement (micron)');
% % % % % NumTicks = 25;
% % % % % L = get(gca,'XLim');
% % % % % set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% % % % % % datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% % % % % datetick('x','HH:MM','keeplimits','keepticks');
% % % % % legend('Pressure','Horizontal shear','Up shear','Normal opening');
% % % % % % xlim([737202.66 737202.675]);
% % % % % grid on;
% % % % % 
% % % % % figure(9611)
% % % % % 
% % % % % % subplot(2,1,2)
% % % % % plot3(S1IEEast(12861:37769),S1IENord(12861:37769),S1IEUp(12861:37769),'r-','linewidth',2);hold on
% % % % % plot3(S1IEEast(37769:48598),S1IENord(37769:48598),S1IEUp(37769:48598),'r:','linewidth',2);hold on
% % % % % plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
% % % % % grid on;
% % % % % xlabel('S1IEEast');
% % % % % ylabel('S1IENord');
% % % % % zlabel('S1IEUp');
% % % % % xlim([-0.5e-4 3e-4]);
% % % % % ylim([-2.5e-4 1.5e-4]);
% % % % % zlim([-3e-4 0.5e-4]);
% % % % % view(90,0);
% % % % % axis square
% % % % % 
% % % % % figure(9612)
% % % % % 
% % % % % % subplot(2,1,2)
% % % % % plot3(S1IEEast(12861:37769),S1IENord(12861:37769),S1IEUp(12861:37769),'r-','linewidth',2);hold on
% % % % % plot3(S1IEEast(37769:48598),S1IENord(37769:48598),S1IEUp(37769:48598),'r:','linewidth',2);hold on
% % % % % plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
% % % % % grid on;
% % % % % xlabel('S1IEEast');
% % % % % ylabel('S1IENord');
% % % % % zlabel('S1IEUp');
% % % % % xlim([-0.5e-4 3e-4]);
% % % % % ylim([-2.5e-4 1.5e-4]);
% % % % % zlim([-3e-4 0.5e-4]);
% % % % % view(0,90);
% % % % % axis square

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Stereoplot Maria Kakurina Routine(s)
Pressure10Hz=Pz1RCfRCf(12861:48598);
% Debit10Hz=Test_372(:,2);

% Northern10Hz = S1ENord(5589:25432);
% Western10Hz = S1EEast(5589:25432); %Attention
% Vertical10Hz = S1EUp(5589:25432);
% Time10Hz=[0:length(Pressure10Hz)-1]'/10;

Northern10Hz = S1IENord(12861:48598);
Western10Hz = S1IEEast(12861:48598); %Attention
Vertical10Hz = S1IEUp(12861:48598);

% Northern10Hz = S1TotNord(12861:48598);
% Western10Hz = S1TotEast(12861:48598)*(1); %Attention
% Vertical10Hz = S1TotUp(12861:48598);

Time10Hz=[0:length(Pressure10Hz)-1]'/10;

% cut data out
% %%---------------------
% ix1=1;%initial
% ix2=1400;%initial
% ix1=1;%before fracture growth at 3184psi
% ix2=5800;%before fracture growth at 3184psi
% ix1=5801;%fracture growth
% ix2=10000;%fracture growth
% ix1=5801;%fracture growth detail at 3184psi 5801
% ix2=6150;%fracture growth detail at 3184psi 6150
% ix1=6151;%fracture growth detail at 3184psi 6151
% ix2=6500;%fracture growth detail at 3184psi 6500
% ix1=6501;%fracture growth detail at 3184psi 6501
% ix2=6850;%fracture growth detail at 3184psi 6850
% ix1=6851;%fracture growth detail at 3184psi 6851
% ix2=7200;%fracture growth detail at 3184psi 7200
% ix1=7201;%fracture growth detail at 3184psi 7201
% ix2=7550;%fracture growth detail at 3184psi 7550
% ix1=7551;%fracture growth detail at 3184psi 7551
% ix2=7900;%fracture growth detail at 3184psi 7900
% ix1=7901;%fracture growth detail at 3184psi 7901
% ix2=8250;%fracture growth detail at 3184psi 8250
% ix1=8251;%fracture growth detail at 3184psi 8251
% ix2=8600;%fracture growth detail at 3184psi 8600
% ix1=8601;%fracture growth detail at 3184psi 8601
% ix2=8950;%fracture growth detail at 3184psi 8950
% ix1=8951;%fracture growth detail at 3184psi 8951
% ix2=9300;%fracture growth detail at 3184psi 9300
% ix1=9301;%fracture growth detail at 3184psi 9301
% ix2=9650;%fracture growth detail at 3184psi 9650
% ix1=9651;%fracture growth detail at 3184psi 9651
% ix2=10000;%fracture growth detail at 3184psi 10000
% ix1=10001;%fracture growth detail at 3184psi 10001
% ix2=10350;%fracture growth detail at 3184psi 10350
% ix1=10351;%fracture growth detail at 3184psi 10351
% ix2=10700;%fracture growth detail at 3184psi 10700
% ix1=10701;%fracture growth detail at 3184psi 10701
% ix2=11050;%fracture growth detail at 3184psi 11050
% ix1=10051;%fracture growth detail at 3184psi 10051
% ix2=10400;%fracture growth detail at 3184psi 11400
% ix1=10201;%fracture growth continued
% ix2=24870;%fracture growth continued
%  ix1=24781;%shuttin
%  ix2=35700;%shuttin
%  ix1=35351;%end shuttin
%  ix2=35700;%end shuttin
ix1=1;%all
ix2=35700;%all
 

Pressure10Hz=Pressure10Hz(ix1:ix2);
% Debit10Hz=Debit10Hz(ix1:ix2);
Western10Hz=Western10Hz(ix1:ix2);
Northern10Hz=Northern10Hz(ix1:ix2);
Vertical10Hz=Vertical10Hz(ix1:ix2);
Time10Hz=Time10Hz(ix1:ix2);

% bring time and displacement to 0 / origin at the start of the serie
Time10Hz=Time10Hz-Time10Hz(1);
Western10Hz=Western10Hz-Western10Hz(1);
Northern10Hz=Northern10Hz-Northern10Hz(1);
Vertical10Hz=Vertical10Hz-Vertical10Hz(1);

%correction effet clamp
figure (94)

plot(Time10Hz,Western10Hz,'bo-','linewidth',2);hold on
plot(Time10Hz,Northern10Hz,'ro-','linewidth',2);hold on
plot(Time10Hz,Vertical10Hz,'gro-','linewidth',2);hold on
xlabel('Time');
ylabel('Disp');
% xlim([-2e-4 6e-4]);
% ylim([-2e-4 6e-4]);
% axis square;
grid on;
title('Disp-vs-time');

figure (95)

plot(Pressure10Hz,Western10Hz,'bo-','linewidth',2);hold on
plot(Pressure10Hz,Northern10Hz,'ro-','linewidth',2);hold on
plot(Pressure10Hz,Vertical10Hz,'gro-','linewidth',2);hold on
xlabel('Pressure');
ylabel('Disp');
% xlim([-2e-4 6e-4]);
% ylim([-2e-4 6e-4]);
% axis square;
grid on;
title('Disp-vs-Press');



a=[Time10Hz,Pressure10Hz,Western10Hz,Northern10Hz,Vertical10Hz];

a=sortrows(a,2);

time_s=a(:,1);
Pressure_s=a(:,2);
Western_s=a(:,3);
Northern_s=a(:,4);
Vertical_s=a(:,5);

%      n= 35;%initial
%      n= 29;%beforefracture growth at 3184psi
%     n= 35;%fracture growth
          n= 35;%fracture growth detail
%      n= 30;%fracture growth continued
%      n= 35;%shuttin
%         n= 35;%end shuttin
        n= 30;%all
 
col=jet(n);
figure('Position',[157          98        1303         845]);
ax(1)=axes('Position',[0.07    0.55    0.3830    0.40]);
axesdefaultformating(ax(1));
hold on 
for i=1:n
    ix = [ length(Time10Hz)/n*(i-1)+1:length(Time10Hz)/n*i ];
    plot(time_s(ix),Pressure_s(ix),'o','Color',col(i,:))
end
grid on
% xlim([0 2000]);
ylim([500 4000]);
xlabel('Time [s]')
ylabel('Pressure [psi]');

ax(2)=axes('Position',[0.07    0.09    0.38    0.40]);
axesdefaultformating(ax(2));
hold on
for i=1:n
    ix=[length(Time10Hz)/n*(i-1)+1:length(Time10Hz)/n*i ];
    plot3(Western_s(ix),Northern_s(ix),Vertical_s(ix),'o','Color',col(i,:))
end
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
view(44,20);
grid on
xlabel('Eastern [m]')
ylabel('Northing [m]');
zlabel('Vertical [m]');
% xlim([-4e-5 14e-5]);
% ylim([-4e-5 14e-5]);
% zlim([-14e-5 4e-5])

We_vec=Western10Hz(2:end)-Western10Hz(1:end-1);
No_vec=Northern10Hz(2:end)-Northern10Hz(1:end-1);
Ve_vec=Vertical10Hz(2:end)-Vertical10Hz(1:end-1);
m=[Pressure10Hz(1:end-1),We_vec,No_vec,Ve_vec];

mm=sortrows(m,1);
z=[0 0 0 0];
mm=[z;mm];

ve=[
        mm(:,2)'
        mm(:,3)'
        mm(:,4)'
        ];  
    
ax(3)=axes('Position',[0.55 0.45 0.35 0.53]);
axesdefaultformating(ax(3));
hold on
stereoframe;
SS=[];
O=[];
normclass=[[0:0.05:0.5] Inf];
markersi=[1 2 3 4 5 6 7 8 10 12 15 20];
linesi  =[0.5 0.5 0.5 0.5 1 1 1 2 2  3  3  4];

for i=1:n;
    ix=[ length(Time10Hz)/n*(i-1)+1:length(Time10Hz)/n*i ];
    v=ve(:,ix);
    for j=1:length(v(1,:)); S(j)=norm(v(:,j)); end;
    SS=[SS S];
    [ dd da ] = l2tp(v(2,:),v(1,:),v(3,:)); dd=dd'; da=da';
    ix=find(da<0); da(ix)=-da(ix); dd(ix)=bringbetween0to360(dd(ix)+180); 
    for j=1:length(normclass)-1
        ix=find(S>=normclass(j) & S<normclass(j+1));        
         h=stereoplot([dd(ix)+180 90-da(ix)],'equalangle','o');%good one
%         h=stereoplot([dd(ix) 90-da(ix)],'equalangle','o');%good one
        set(h,'MarkerSize',markersi(j),'LineWidth',linesi(j),'Color',col(i,:))
    end
     O=[O; dd+180 90-da];%good one
%     O=[O; dd 90-da];
end

ax(3)=axes('Position',[0.55 0.01 0.35 0.43]);
axesdefaultformating(ax(3));
hold on
stereocontouring([O SS'],'equalangle',1/50)
stereoframe;

%Figure pour le papier ARMA
figure(1313)

subplot (2,2,1)
plot(Pz1RCfRCf(12861:37769)-Pz1RCfRCf(726),sqrt((S1YatesIE(12861:37769)-S1YatesIE(12861)).^2+(S1TopIE(12861:37769)-S1TopIE(12861)).^2),'c-','linewidth',3);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),sqrt((S1YatesIE(37769:48598)-S1YatesIE(12861)).^2+(S1TopIE(37769:48598)-S1TopIE(12861)).^2),'c:','linewidth',3);hold on

plot(Pz1RCfRCf(12861:37769)-Pz1RCfRCf(726),(S1WellAxialIE(12861:37769)-S1WellAxialIE(12861)),'b-','linewidth',1);hold on
plot(Pz1RCfRCf(37769:48598)-Pz1RCfRCf(726),(S1WellAxialIE(37769:48598)-S1WellAxialIE(12861)),'b:','linewidth',1);hold on

xlim([500 4000]);
ylim([-1e-4 3.5e-4]);
grid on;
% xlabel('Chamber Pressure psi');
ylabel('Normalized Displacements');
% title ('Pressure psi');
legend('Radial','Radial shut in','Axial','Axial shut in');

subplot (2,2,2)
plot3(S1IEEast(12861:37769),S1IENord(12861:37769),S1IEUp(12861:37769),'r-','linewidth',2);hold on
plot3(S1IEEast(37769:48598),S1IENord(37769:48598),S1IEUp(37769:48598),'r--','linewidth',2);hold on
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
grid on;
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
xlim([-3e-4 2e-4]);
ylim([-4e-4 1e-4]);
zlim([-4e-4 1e-4]);
view(90,0);
axis square

subplot(2,2,3)
plot(QuizixP(1:6006),QuizixQ(1:6006),'k-','linewidth',2);hold on
xlim([500 4000]);
ylim([-0.01 0.5]);
grid on
xlabel('Chamber Pressure psi');
ylabel('Flowrate (mL/Min)');

subplot(2,2,4)
plot3(S1IEEast(12861:37769),S1IENord(12861:37769),S1IEUp(12861:37769),'r-','linewidth',2);hold on
plot3(S1IEEast(37769:48598),S1IENord(37769:48598),S1IEUp(37769:48598),'r--','linewidth',2);hold on
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
grid on;
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
xlim([-3e-4 2e-4]);
ylim([-4e-4 1e-4]);
zlim([-4e-4 1e-4]);
view(0,90);
axis square

figure(1314)

subplot (2,1,1)
plot3(S1IEEast(12861:14572),S1IENord(12861:14572),S1IEUp(12861:14572),'k-','linewidth',2);hold on
plot3(S1IEEast(14572:22108),S1IENord(14572:22108),S1IEUp(14572:22108),'c-','linewidth',2);hold on
plot3(S1IEEast(22108:37769),S1IENord(22108:37769),S1IEUp(22108:37769),'r-','linewidth',2);hold on
plot3(S1IEEast(37769:48598),S1IENord(37769:48598),S1IEUp(37769:48598),'b--','linewidth',2);hold on
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
grid on;
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
xlim([-0.5e-4 3e-4]);
ylim([-2.5e-4 1.5e-4]);
zlim([-3e-4 0.5e-4]);
view(90,0);
axis square

subplot(2,1,2)
plot3(S1IEEast(12861:14572),S1IENord(12861:14572),S1IEUp(12861:14572),'k-','linewidth',2);hold on
plot3(S1IEEast(14572:22108),S1IENord(14572:22108),S1IEUp(14572:22108),'c-','linewidth',2);hold on
plot3(S1IEEast(22108:37769),S1IENord(22108:37769),S1IEUp(22108:37769),'r-','linewidth',2);hold on
plot3(S1IEEast(37769:48598),S1IENord(37769:48598),S1IEUp(37769:48598),'b--','linewidth',2);hold on
plot3(xbore,ybore,zbore,'gr-','linewidth',3);hold on
grid on;
xlabel('S1IEEast');
ylabel('S1IENord');
zlabel('S1IEUp');
xlim([-0.5e-4 3e-4]);
ylim([-2.5e-4 1.5e-4]);
zlim([-3e-4 0.5e-4]);
view(0,90);
axis square

