load('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/FSBTOMO/Baseline6.mat')
load('/Volumes/FSC_Data/FSC_CASSM/May2and3data/20230502080410/dsiDataV1.mat')


sx = []; sy = []; sz = []; rx = []; ry = []; rz = [];
for gath = 1:24
        sx  = [sx, dsiDataV1.th{gath}(31,1:44)];%eastings
        sy  = [sy, dsiDataV1.th{gath}(29,1:44)];%northings
        sz  = [sz, dsiDataV1.th{gath}(33,1:44)];
        rx  = [rx, dsiDataV1.th{gath}(37,1:44)];%eastings
        ry  = [ry, dsiDataV1.th{gath}(35,1:44)];%northings
        rz  = [rz, dsiDataV1.th{gath}(39,1:44)];
end
badSources = [((18-1)*44+1):18*44 ((23-1)*44+1):23*44];
sx(badSources) = [];
sy(badSources) = [];
sz(badSources) = [];
rx(badSources) = [];
ry(badSources) = [];
rz(badSources) = [];

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

g        = [];
g        = grid3drcfs(outstruct.par,4);
[tt, rays, G] = g.raytrace(outstruct.slow{end,1}, Tx, Rx); %raytracing

save('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/kernel.mat','G')
