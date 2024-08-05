% clear
% clc
% close all
clear instruct
load('/Volumes/FSC_Data/FSC_CASSM/May2and3data/20230502080410/dsiDataV1.mat')
load('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/TravelTimePicksFSC.mat')

%   Regularization + modeling parameter arguments
instruct.regX       = 1;
instruct.regY       = 1;
instruct.regZ       = 1;

%   Non-Linear Algorithm Arguments
%   ------------------------------------------------
instruct.nlres      = 1e-6;
instruct.reslim     = 1e-6;

dataSet = [];

sx = []; sy = []; sz = []; rx = []; ry = []; rz = [];
instruct.data = [];
for gath = 1:24
        sx  = [sx, dsiDataV1.th{gath}(31,1:44)];%eastings
        sy  = [sy, dsiDataV1.th{gath}(29,1:44)];%northings
        sz  = [sz, dsiDataV1.th{gath}(33,1:44)];
        rx  = [rx, dsiDataV1.th{gath}(37,1:44)];%eastings
        ry  = [ry, dsiDataV1.th{gath}(35,1:44)];%northings
        rz  = [rz, dsiDataV1.th{gath}(39,1:44)];
        instruct.data = data(:,1);
end

% removing data and locations for bad sources. 18 and 23
badSources = [((18-1)*44+1):18*44 ((23-1)*44+1):23*44];
sx(badSources) = [];
sy(badSources) = [];
sz(badSources) = [];
rx(badSources) = [];
ry(badSources) = [];
rz(badSources) = [];
instruct.data(badSources) = [];

% Translating the geometry to a local reference frame. 
dataSet(:,1) = sx';
dataSet(:,2) = sy';
dataSet(:,3) = sz';
dataSet(:,4) = rx';
dataSet(:,5) = ry';
dataSet(:,6) = rz';


[dataSet,tx1,ty1,tz1] = medianTranslate(dataSet); % First translation

p1     = pca(dataSet(:,1:3));
norml1 = p1(:,3);
vectz = [0; 0; 1];
vectx = [1; 0; 0];
vecty = [0; 1; 0];
angle1 = cross(norml1,vectz);
Costheta1 = max(min(dot(angle1,vectx)/(norm(angle1)*norm(vectx)),1),-1);
theta1 = real(acosd(Costheta1));
rot1 = -(theta1 + 90)*pi/180;
dataSet = rotateDataset(0,0,rot1,dataSet); % First rotation

[dataSet,tx2,ty2,tz2] = medianTranslate(dataSet); % Second translation

p2     = pca(dataSet(:,1:3));
norml2 = p2(:,3);
angle2 = cross(norml2,vecty);
Costheta2 = max(min(dot(angle2,vectz)/(norm(angle2)*norm(vectz)),1),-1);
theta2 = real(acosd(Costheta2));
rot2 = -theta2*pi/180;
dataSet = rotateDataset(0,rot2,0,dataSet); % Second rotation

[dataSet,tx3,ty3,tz3] = medianTranslate(dataSet); % Third translation

figure
miniPlotPicks(dataSet);
axis equal tight


Tx = dataSet(:,1:3);
Rx = dataSet(:,4:6);


%   (mandatory survey/geometry parameter arguments)
%   ------------------------------------------------
instruct.xsp    = Tx(:,1);
instruct.ysp    = Tx(:,2);
instruct.zsp    = Tx(:,3);
instruct.xrp    = Rx(:,1);
instruct.yrp    = Rx(:,2);
instruct.zrp    = Rx(:,3);
instruct.deltaX = 0.5; 
instruct.deltaY = 0.5;
instruct.deltaZ = 0.5;
instruct.kapX   = 1;
instruct.kapY   = 1;
instruct.kapZ   = 1;
instruct.maxiter = 10;
instruct.relStep = 0.2;
instruct.slow    = [];
instruct.Weight = [];

[outstruct]  = tomo3DTTCR(instruct);

par = outstruct.par;
par.ymin = 0;
par.zmin = 0;
figure
    hold on
    Vp = outstruct.vel{end};
    [M,X,Y] = plotslice3D(Vp,par,'Z-Y',0.5,1,'V_P (m/s)',[]);
    ylabel('Distance along dip direction (m)')
    xlabel('Distance along strike (m)')
    hold off
    set(gca,'FontName','Helvetica')
    set(gca,'FontSize',10)
    colorbar
    patch([0.5 28 33.5 33 20.5 12],[14.5 12 40 56 54.5 42.5],'k','FaceColor','none','EdgeColor','k')
    set(gca,'Layer','top')
    c = max(M,[],'all');