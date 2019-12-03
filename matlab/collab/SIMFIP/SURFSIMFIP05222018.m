% Matlab Routine for reading mHPP Data
% Implementation done by  Y. Guglielmi/J.Durand
%-----------------------------------------------18032015(JayVersion)forTournemire-----------------------------------------------
clear all
close all
clc

%----------SIMFIP----------
%----------Declaration----------
%
rep = '/home/chet/SIMFIP/SURF/SURF_Notch142ft/5-22-2018';
ext = '*.csv';
chemin = fullfile(rep,ext);
list = dir(chemin);

TimeString=[];
TimeGString=[];
TimeD = [];
data0=[];

%All Test
n1=2;
n2=37;
% n1=19;
% n2=37;

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
    if exist('bigdata1');
       bigdata1 = [bigdata1 ; C{UsedColumns}];
    else
       bigdata1 = [C{UsedColumns}];
    end
%        linedata=linedata{1};
%        dataline = str2num(linedata(2:length(linedata)))
            
%        nextline = fgetl(fid);
    [nsamples nc]=size (bigdata1);
    if nsamples > 10000
        nsamples
        datablock = [];
        for i=1:ncol
            datablock = [datablock decimate(bigdata1(:,i),DeciBy)];
        end
        data0 = [data0 ; datablock];
        clear bigdata1;
        
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
Pz1 = data0(:,35);
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
% SIMFIPSURF05222018brut10Hz = [TimeD,micTime,A1,B1,C1,D1,E1,F1,oneT1,Pznm1,Tznm1,microSA1,microSB1,microSC1,microSD1,microSE1,microSF1,microS1T1,microSPznm1,microSTznm1,Fox1,Foy1,Foz1,Tox1,Toy1,Toz1,Dx1,Dy1,Dz1,Rx1,T1,Dxt1,Dyt1,Rxt1,Ryt1,Pz1,Tz1,Pbnm1,Tbnm1,Pb1,Tb1,A2,B2,C2,D2,E2,F2,OneT2,Pznm2,Tznm2,microSA2,microSB2,microSC2,microSD2,microSE2,microSF2,microS1T2,microSPznm2,microSTznm2,Fox2,Foy2,Foz2,Tox2,Toy2,Toz2,Dx2,Dy2,Dz2,Rx2,T2,Dxt2,Dyt2,Rxt2,Ryt2,Pz2,Tz2,Pbnm2,Tbnm2,Up1,Bp1,Up2,Bp2,Pt1,Pt2,Flow];
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
0 -0.000114489 0 0.001625984 0 -9.53402e-5;
0.000114744 0 7.56304e-07 0 0.001628276 0;
0 -7.63944e-7 0 -0.000139802 0 0.02293582];

% Calculate the forces and moments for three principle directions
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
angS1Z = pi*(-50/180); % The angle (0 < angZ < 360� )is around Oz  
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
rotatS1 = rS1z * calculS1;
S1Yates(ni,1) = rotatS1(1,1);
S1Top(ni,1) = rotatS1(2,1);
S1WellAxial(ni,1) = rotatS1(3,1);
end

% correction clamps

calculS1c = [];
S1Yatesc = [];
S1Topc = [];
S1WellAxialc = [];


for ni = 1:LS1
% calculS1c =  [ux1(ni,1)-(-0.00025719+(-17.2e-6*Pz1(ni,1))/300);uy1(ni,1)-(-0.00039795+(23.0e-6*Pz1(ni,1))/300);uz1(ni,1)-(0.000027204+(-6.01e-6*Pz1(ni,1))/300)];  
calculS1c =  [-17.2e-6*Pz1(ni,1)/300;23.0e-6*Pz1(ni,1)/300;-6.01e-6*Pz1(ni,1)/300];  
rotatS1c = rS1z*calculS1c;
S1Yatesc(ni,1) = rotatS1c(1,1);
S1Topc(ni,1) = rotatS1c(2,1);
S1WellAxialc(ni,1) = rotatS1c(3,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Enregistrement en .mat et en .txt
% 
% SIMFIPSURF05212018 = [A1(),B1(),C1(),D1(),E1(),F1(),OneT1(),microSA(),microSB(),microSC(),microSD(),microSE(),microSF(),microSOneT(),Fx(),Fy(),Fz(),Tx(),Ty(),Tz(),Dx(),Dy(),Dz(),Rx(),Ry(),Rz()];
% save('SIMFIPSURF05212018.txt','SIMFIPSURF05212018', '-ASCII');
% save ('SIMFIPSURF05212018.mat');
% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure (1)
subplot(3,1,1)
%plot(TimeD,Flow,'k-','linewidth',2);hold on
plot(TimeD,Flow,'b-','linewidth',2);hold on
grid on;
NumTicks = 10;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
%xlim([0 2100]);
%ylim([-1 40]);
%xlabel('Date');
ylabel('Flow (mL/min)');
title ('Test May 22, 2018');

subplot(3,1,2)
%plot(TimeD,Flow,'k-','linewidth',2);hold on
plot(TimeD,Pz1,'b-','linewidth',2);hold on
plot(TimeD,Up1,'r-','linewidth',2);hold on
plot(TimeD,Bp1,'gr-','linewidth',2);hold on
grid on;
NumTicks = 10;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
datetick('x','dd/mm HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
%xlim([0 2100]);
%ylim([-1 40]);
%xlabel('Date');
ylabel('Pressure (psi)');
legend('Interval Pressure','Top Packer pressure','Bottom Packer Pressure');
% title ('Reference sensor');

%figure (2)
subplot(3,1,3)
plot(TimeD,S1Yates()-S1Yates(1),'b-','linewidth',2);hold on
plot(TimeD,S1Top()-S1Top(1),'r-','linewidth',2);hold on
plot(TimeD,S1WellAxial()-S1WellAxial(1),'gr-','linewidth',2);hold on
NumTicks = 10;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
datetick('x','dd/mm HH:MM','keeplimits','keepticks');
%xlim([0 2100]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('X-Yates','Y-Top','Z-Well Axial');
% title ('Reference sensor');

figure (2)
subplot(3,1,1)
%plot(TimeD,Flow,'k-','linewidth',2);hold on
plot(TimeD,Flow,'b-','linewidth',2);hold on
grid on;
NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
xlim([737202.475 737202.526]);
ylim([-100 3000]);
%xlabel('Date');
ylabel('Flow (mL/min)');
title ('Test May 22, 2018');

subplot(3,1,2)
%plot(TimeD,Flow,'k-','linewidth',2);hold on
plot(TimeD,Pz1,'b-','linewidth',2);hold on
plot(TimeD,Up1,'r-','linewidth',2);hold on
plot(TimeD,Bp1,'gr-','linewidth',2);hold on
grid on;
NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
xlim([737202.475 737202.526]);
ylim([-30 6000]);
%xlabel('Date');
ylabel('Pressure (psi)');
legend('Interval Pressure','Top Packer pressure','Bottom Packer Pressure');
% title ('Reference sensor');

%figure (2)
subplot(3,1,3)
% plot(TimeD,S1Yates()-S1Yates(1),'b-','linewidth',2);hold on
% plot(TimeD,S1Top()-S1Top(1),'r-','linewidth',2);hold on
% plot(TimeD,(S1WellAxial()-S1WellAxial(1))*10,'gr-','linewidth',2);hold on
plot(TimeD,ux1()-ux1(1),'b-','linewidth',2);hold on
plot(TimeD,uy1()-uy1(1),'r-','linewidth',2);hold on
plot(TimeD,(uz1()-uz1(1))*10,'gr-','linewidth',2);hold on

NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
% legend('X-Yates','Y-Top','Z-Well Axial *10');
legend('Ux','Uy','Uz *10');
% title ('Reference sensor');

figure (21)
% subplot(3,1,3)
% plot(TimeD,S1Yates()-S1Yates(1),'b-','linewidth',2);hold on
% plot(TimeD,S1Top()-S1Top(1),'r-','linewidth',2);hold on
% plot(TimeD,(S1WellAxial()-S1WellAxial(1))*10,'gr-','linewidth',2);hold on
plot(TimeD,ux1()-ux1(1),'b-','linewidth',2);hold on
plot(TimeD,uy1()-uy1(1),'r-','linewidth',2);hold on
plot(TimeD,(uz1()-uz1(1))*10,'gr-','linewidth',2);hold on
plot(TimeD,10*Pz1/1e8,'c-','linewidth',2);hold on
NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
% legend('X-Yates','Y-Top','Z-Well Axial *10');
legend('Ux','Uy','Uz *10','Chamber Pressure');
% title ('Reference sensor');

figure (3)
subplot(3,1,1)
plot(TimeD,ux1()-ux1(1),'b-','linewidth',2);hold on
plot(TimeD,(-7.8e-6*Pz1/300)-0.00028805,'b:','linewidth',2);hold on
plot(TimeD,(-17.2e-6*Pz1/300)-0.00025719,'b--','linewidth',2);hold on
grid on;
NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
% xlim([737202.4829 737202.4846]);
xlim([737202.475 737202.526]);
%ylim([-1 40]);
%xlabel('Date');
ylabel('Pressure (psi)');
legend('Ux1 measurement','Ux1 clamp low','Ux1 clamp high');
 title ('22 05 2018 Details Clamping');

subplot(3,1,2)
plot(TimeD,uy1()-uy1(1),'r-','linewidth',2);hold on
plot(TimeD,-0.00035549+(10.0e-6*Pz1/300),'r:','linewidth',2);hold on
plot(TimeD,-0.00039795+(23.0e-6*Pz1/300),'r--','linewidth',2);hold on

NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.4829 737202.4846]);
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('Uy1 measurement','Uy1 clamp low','Uy1 clamp high');
% title ('Reference sensor');


subplot(3,1,3)
plot(TimeD,(uz1()-uz1(1)),'gr-','linewidth',2);hold on
plot(TimeD,0.0000152+(-2.31e-6*Pz1/300),'gr:','linewidth',2);hold on
plot(TimeD,0.000027204+(-6.01e-6*Pz1/300),'gr--','linewidth',2);hold on
NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.4829 737202.4846]);
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('Uz1 measurement','Uz1 clamp low','Uz1 clamp high');
% title ('Reference sensor');

figure (4)
subplot(3,1,1)
plot(TimeD,ux1()-ux1(1),'b-','linewidth',2);hold on
% plot(TimeD,(-7.8e-6*Pz1/300)-0.00028805,'b:','linewidth',2);hold on
plot(TimeD,ux1()-ux1(1)-((-17.2e-6*Pz1/300)),'b--','linewidth',2);hold on
grid on;
NumTicks = 400;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% datetick('x','dd/mm HH:MM');
% xlim([737202.4829 737202.4846]);
xlim([737202.475 737202.526]);
%ylim([-1 40]);
%xlabel('Date');
ylabel('Pressure (psi)');
legend('Ux1 measurement','Ux1 clamp high');
 title ('22 05 2018 Details Clamping');

subplot(3,1,2)
plot(TimeD,uy1()-uy1(1),'r-','linewidth',2);hold on
% plot(TimeD,-0.00035549+(10.0e-6*Pz1/300),'r:','linewidth',2);hold on
plot(TimeD,uy1()-uy1(1)-((23.0e-6*Pz1/300)),'r--','linewidth',2);hold on

NumTicks = 400;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.4829 737202.4846]);
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('Uy1 measurement','Uy1 clamp high');
% title ('Reference sensor');


subplot(3,1,3)
plot(TimeD,(uz1()-uz1(1)),'gr-','linewidth',2);hold on
% plot(TimeD,0.0000152+(-2.31e-6*Pz1/300),'gr:','linewidth',2);hold on
plot(TimeD,(uz1()-uz1(1))-((-6.01e-6*Pz1/300)),'gr--','linewidth',2);hold on
NumTicks = 400;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
% xlim([737202.4829 737202.4846]);
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('Uz1 measurement','Uz1 clamp high');
% title ('Reference sensor');

figure(5)

subplot(3,1,1)
% plot(TimeD,Dx1()-Dx1(1),'b--','linewidth',2);hold on
plot(TimeD,S1Yates()-S1Yates(1),'b-','linewidth',2);hold on
plot(TimeD,S1Yatesc()-S1Yatesc(1),'b:','linewidth',2);hold on

NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('X-Yates','X-Yates c ');
% title ('Reference sensor');

subplot(3,1,2)

% plot(TimeD,Dy1()-Dy1(1),'r--','linewidth',2);hold on
plot(TimeD,S1Top()-S1Top(1),'r-','linewidth',2);hold on
plot(TimeD,S1Topc()-S1Topc(1),'r:','linewidth',2);hold on

NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('Y-Top','Y-Top c ');
% title ('Reference sensor');

subplot(3,1,3)

% plot(TimeD,Dz1()-Dz1(1),'gr--','linewidth',2);hold on
plot(TimeD,S1WellAxial()-S1WellAxial(1),'gr-','linewidth',2);hold on
plot(TimeD,S1WellAxialc()-S1WellAxialc(1),'gr:','linewidth',2);hold on
NumTicks = 120;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.475 737202.526]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('Z-Well Axial','Z-Well Axial c ');
% title ('Reference sensor');

figure(6)

subplot(3,1,1)

plot(TimeD,S1Yates()-S1Yates(1)-(S1Yatesc()-S1Yatesc(1)),'b-','linewidth',2);hold on

NumTicks = 200;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.495 737202.5205]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('X-Yates','X-Yates c ');
% title ('Reference sensor');

subplot(3,1,2)

plot(TimeD,S1Top()-S1Top(1)-(S1Topc()-S1Topc(1)),'r-','linewidth',2);hold on

NumTicks = 200;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.495 737202.5205]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('Y-Top','Y-Top c ');
% title ('Reference sensor');

subplot(3,1,3)


plot(TimeD,S1WellAxial()-S1WellAxial(1)-(S1WellAxialc()-S1WellAxialc(1)),'gr-','linewidth',2);hold on

NumTicks = 200;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.495 737202.5205]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('Z-Well Axial','Z-Well Axial c ');
% title ('Reference sensor');

figure(7)

subplot(3,1,1)

plot(TimeD,Flow,'b-','linewidth',2);hold on
NumTicks = 200;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.495 737202.5205]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
 ylabel('Flowrate (mL/min)');
% legend('X-Yates','X-Yates c ');
 title ('Displacements corrected from Clamp effect');

subplot(3,1,2)

plot(TimeD,Pz1,'b-','linewidth',2);hold on
plot(TimeD,Up1,'r-','linewidth',2);hold on
plot(TimeD,Bp1,'gr-','linewidth',2);hold on
NumTicks = 200;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.495 737202.5205]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Pressure (psi)');
legend('Chamber','Top Packer','Bottom Packer');
% title ('Reference sensor');

subplot(3,1,3)

plot(TimeD,S1Yates()-S1Yates(1)-(S1Yatesc()-S1Yatesc(1))-0.0001404,'b-','linewidth',2);hold on
plot(TimeD,S1Top()-S1Top(1)-(S1Topc()-S1Topc(1))+0.0004473,'r-','linewidth',2);hold on
plot(TimeD,S1WellAxial()-S1WellAxial(1)-(S1WellAxialc()-S1WellAxialc(1))-2.746e-5,'gr-','linewidth',2);hold on

NumTicks = 200;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.495 737202.5205]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
ylabel('Displacement (micron)');
legend('X-Yates','Y-Top','Z-Well Axial');
% title ('Reference sensor');

figure(71)

plot(TimeD(11036:13601),(Pz1(11036:13601)-Pz1(11036)),'k-','linewidth',2);hold on
plot(TimeD(13601:13631),(Pz1(13601:13631)-Pz1(11036)),'c-','linewidth',2);hold on
plot(TimeD(13631:13754),(Pz1(13631:13754)-Pz1(11036)),'r-','linewidth',2);hold on
plot(TimeD(13754:15064),(Pz1(13754:15064)-Pz1(11036)),'b-','linewidth',2);hold on
plot(TimeD(15064:22516),(Pz1(15064:22516)-Pz1(11036)),'gr-','linewidth',2);hold on
% plot(TimeD(),S1Yates()-S1Yates(1)-(S1Yatesc()-S1Yatesc(1))-0.0001404,'b-','linewidth',2);hold on
% plot(TimeD(),S1Top()-S1Top(1)-(S1Topc()-S1Topc(1))+0.0004473,'r-','linewidth',2);hold on
% plot(TimeD(),S1WellAxial()-S1WellAxial(1)-(S1WellAxialc()-S1WellAxialc(1))-2.746e-5,'gr-','linewidth',2);hold on

NumTicks = 400;
L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks));
% datetick('x','dd/mm HH:MM','keeplimits','keepticks');
datetick('x','HH:MM','keeplimits','keepticks');
xlim([737202.5003896169 737202.518]);
%ylim([-1 40]);
grid on;
%xlabel('Date');
% legend('Pressure (psi)','X-Yates','Y-Top','Z-Well Axial');
% legend('Chamber','Top Packer','Bottom Packer');
title ('Pressure psi');


figure(8)

plot3(S1Yates(11036:13601)-S1Yates(11036)-(S1Yatesc(11036:13601)-S1Yatesc(11036)),S1Top(11036:13601)-S1Top(11036)-(S1Topc(11036:13601)-S1Topc(11036)),S1WellAxial(11036:13601)-S1WellAxial(11036)-(S1WellAxialc(11036:13601)-S1WellAxialc(11036)),'b-','linewidth',2);hold on
plot3(S1Yates(13601:13631)-S1Yates(11036)-(S1Yatesc(13601:13631)-S1Yatesc(11036)),S1Top(13601:13631)-S1Top(11036)-(S1Topc(13601:13631)-S1Topc(11036)),S1WellAxial(13601:13631)-S1WellAxial(11036)-(S1WellAxialc(13601:13631)-S1WellAxialc(11036)),'c-','linewidth',2);hold on
plot3(S1Yates(13631:13754)-S1Yates(11036)-(S1Yatesc(13631:13754)-S1Yatesc(11036)),S1Top(13631:13754)-S1Top(11036)-(S1Topc(13631:13754)-S1Topc(11036)),S1WellAxial(13631:13754)-S1WellAxial(11036)-(S1WellAxialc(13631:13754)-S1WellAxialc(11036)),'r-','linewidth',2);hold on
plot3(S1Yates(13754:15064)-S1Yates(11036)-(S1Yatesc(13754:15064)-S1Yatesc(11036)),S1Top(13754:15064)-S1Top(11036)-(S1Topc(13754:15064)-S1Topc(11036)),S1WellAxial(13754:15064)-S1WellAxial(11036)-(S1WellAxialc(13754:15064)-S1WellAxialc(11036)),'b-','linewidth',2);hold on
plot3(S1Yates(15064:22516)-S1Yates(11036)-(S1Yatesc(15064:22516)-S1Yatesc(11036)),S1Top(15064:22516)-S1Top(11036)-(S1Topc(15064:22516)-S1Topc(11036)),S1WellAxial(15064:22516)-S1WellAxial(11036)-(S1WellAxialc(15064:22516)-S1WellAxialc(11036)),'gr-','linewidth',2);hold on

xlim([-5e-4 5e-4]);
ylim([-5e-4 5e-4]);
zlim([-5e-4 5e-4]);
grid on;
xlabel('S1Yates');
ylabel('S1Top');
zlabel('S1WellAxis');
axis square
% title ('Reference sensor');

figure(9)

plot(S1WellAxial(11036:13601)-S1WellAxial(11036)-(S1WellAxialc(11036:13601)-S1WellAxialc(11036)),S1Top(11036:13601)-S1Top(11036)-(S1Topc(11036:13601)-S1Topc(11036)),'b-','linewidth',2);hold on
plot(S1WellAxial(13601:13631)-S1WellAxial(11036)-(S1WellAxialc(13601:13631)-S1WellAxialc(11036)),S1Top(13601:13631)-S1Top(11036)-(S1Topc(13601:13631)-S1Topc(11036)),'c-','linewidth',2);hold on
plot(S1WellAxial(13631:13754)-S1WellAxial(11036)-(S1WellAxialc(13631:13754)-S1WellAxialc(11036)),S1Top(13631:13754)-S1Top(11036)-(S1Topc(13631:13754)-S1Topc(11036)),'r-','linewidth',2);hold on
plot(S1WellAxial(13754:15064)-S1WellAxial(11036)-(S1WellAxialc(13754:15064)-S1WellAxialc(11036)),S1Top(13754:15064)-S1Top(11036)-(S1Topc(13754:15064)-S1Topc(11036)),'b-','linewidth',2);hold on
plot(S1WellAxial(15064:22516)-S1WellAxial(11036)-(S1WellAxialc(15064:22516)-S1WellAxialc(11036)),S1Top(15064:22516)-S1Top(11036)-(S1Topc(15064:22516)-S1Topc(11036)),'gr-','linewidth',2);hold on

xlim([-5e-4 5e-4]);
ylim([-5e-4 5e-4]);
% zlim([-5e-4 5e-4]);
grid on;
xlabel('S1WellAxis');
ylabel('S1Top');
axis square
title ('Vertical-Top plane');

figure(10)

plot(S1Yates(11036:13601)-S1Yates(11036)-(S1Yatesc(11036:13601)-S1Yatesc(11036)),S1Top(11036:13601)-S1Top(11036)-(S1Topc(11036:13601)-S1Topc(11036)),'b-','linewidth',2);hold on
plot(S1Yates(13601:13631)-S1Yates(11036)-(S1Yatesc(13601:13631)-S1Yatesc(11036)),S1Top(13601:13631)-S1Top(11036)-(S1Topc(13601:13631)-S1Topc(11036)),'c-','linewidth',2);hold on
plot(S1Yates(13631:13754)-S1Yates(11036)-(S1Yatesc(13631:13754)-S1Yatesc(11036)),S1Top(13631:13754)-S1Top(11036)-(S1Topc(13631:13754)-S1Topc(11036)),'r-','linewidth',2);hold on
plot(S1Yates(13754:15064)-S1Yates(11036)-(S1Yatesc(13754:15064)-S1Yatesc(11036)),S1Top(13754:15064)-S1Top(11036)-(S1Topc(13754:15064)-S1Topc(11036)),'b-','linewidth',2);hold on
plot(S1Yates(15064:22516)-S1Yates(11036)-(S1Yatesc(15064:22516)-S1Yatesc(11036)),S1Top(15064:22516)-S1Top(11036)-(S1Topc(15064:22516)-S1Topc(11036)),'gr-','linewidth',2);hold on

xlim([-5e-4 5e-4]);
ylim([-5e-4 5e-4]);
grid on;
xlabel('S1Yates');
ylabel('S1Top');
axis square
title ('Yates - Top');

figure(11)

plot(S1WellAxial(11036:13601)-S1WellAxial(11036)-(S1WellAxialc(11036:13601)-S1WellAxialc(11036)),S1Yates(11036:13601)-S1Yates(11036)-(S1Yatesc(11036:13601)-S1Yatesc(11036)),'b-','linewidth',2);hold on
plot(S1WellAxial(13601:13631)-S1WellAxial(11036)-(S1WellAxialc(13601:13631)-S1WellAxialc(11036)),S1Yates(13601:13631)-S1Yates(11036)-(S1Yatesc(13601:13631)-S1Yatesc(11036)),'c-','linewidth',2);hold on
plot(S1WellAxial(13631:13754)-S1WellAxial(11036)-(S1WellAxialc(13631:13754)-S1WellAxialc(11036)),S1Yates(13631:13754)-S1Yates(11036)-(S1Yatesc(13631:13754)-S1Yatesc(11036)),'r-','linewidth',2);hold on
plot(S1WellAxial(13754:15064)-S1WellAxial(11036)-(S1WellAxialc(13754:15064)-S1WellAxialc(11036)),S1Yates(13754:15064)-S1Yates(11036)-(S1Yatesc(13754:15064)-S1Yatesc(11036)),'b-','linewidth',2);hold on
plot(S1WellAxial(15064:22516)-S1WellAxial(11036)-(S1WellAxialc(15064:22516)-S1WellAxialc(11036)),S1Yates(15064:22516)-S1Yates(11036)-(S1Yatesc(15064:22516)-S1Yatesc(11036)),'gr-','linewidth',2);hold on

xlim([-5e-4 5e-4]);
ylim([-5e-4 5e-4]);
zlim([-5e-4 5e-4]);
grid on;
xlabel('S1wellAxis');
ylabel('S1Yates');
axis square
title ('Vertical - Yates');

figure(12)

% subplot(3,1,1)

plot(Pz1(11036:13601)-Pz1(11036),S1WellAxial(11036:13601)-S1WellAxial(11036)-(S1WellAxialc(11036:13601)-S1WellAxialc(11036)),'b-','linewidth',2);hold on
plot(Pz1(13601:15064)-Pz1(11036),S1WellAxial(13601:15064)-S1WellAxial(11036)-(S1WellAxialc(13601:15064)-S1WellAxialc(11036)),'r-','linewidth',2);hold on
plot(Pz1(15064:22516)-Pz1(11036),S1WellAxial(15064:22516)-S1WellAxial(11036)-(S1WellAxialc(15064:22516)-S1WellAxialc(11036)),'gr-','linewidth',2);hold on
xlim([-5 4005]);
ylim([-2e-5 12e-5]);
grid on;
xlabel('Chamber Pressure psi');
ylabel('S1WellAxial micrometer');
 title ('Displacement vs Pressure');

figure(13)

% subplot(3,1,2)

plot(Pz1(11036:13601)-Pz1(11036),S1Yates(11036:13601)-S1Yates(11036)-(S1Yatesc(11036:13601)-S1Yatesc(11036)),'b-','linewidth',2);hold on
plot(Pz1(13601:15064)-Pz1(11036),S1Yates(13601:15064)-S1Yates(11036)-(S1Yatesc(13601:15064)-S1Yatesc(11036)),'r-','linewidth',2);hold on
plot(Pz1(15064:22516)-Pz1(11036),S1Yates(15064:22516)-S1Yates(11036)-(S1Yatesc(15064:22516)-S1Yatesc(11036)),'gr-','linewidth',2);hold on
xlim([-5 4005]);
ylim([-3.5e-4 0.5e-4]);
grid on;
xlabel('Chamber Pressure psi');
ylabel('S1Yates micrometer');
 title ('Displacement vs Pressure');

figure(14)
 
% subplot(3,1,3)

plot(Pz1(11036:13601)-Pz1(11036),S1Top(11036:13601)-S1Top(11036)-(S1Topc(11036:13601)-S1Topc(11036)),'b-','linewidth',2);hold on
plot(Pz1(13601:15064)-Pz1(11036),S1Top(13601:15064)-S1Top(11036)-(S1Topc(13601:15064)-S1Topc(11036)),'r-','linewidth',2);hold on
plot(Pz1(15064:22516)-Pz1(11036),S1Top(15064:22516)-S1Top(11036)-(S1Topc(15064:22516)-S1Topc(11036)),'gr-','linewidth',2);hold on
xlim([-5 4005]);
ylim([-1e-4 3.5e-4]);
grid on;
xlabel('Chamber Pressure psi');
ylabel('S1Top micrometer');
 title ('Displacement vs Pressure');

stop

