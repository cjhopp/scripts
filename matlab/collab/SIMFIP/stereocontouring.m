function varargout=stereocontouring(D,projection,ccsize,meanorient)
% stereocontouring(D,projection,ccsize,meanorient)
% h=stereocontouring(D,projection,ccsize,meanorient)
% [h a p t]=stereocontouring(D,projection,ccsize);
% [h ht]=stereocontouring(D,projection,ccsize);
% [h ht a p t]=stereocontouring(D,projection,ccsize);
% D=[azim pend weigth]   n x 3 matrix or n x 2 matrix
%       if weight not given, weight=1 for all data
% projection: 'equalangle' {default} or 'equalarea'
% ccsize: counting circle size in relation to the entire hemisphere
%         {default}  1/50
% meanorient: if meanorient=='yes', the mean orientation is computed and
% displayed {default, meanorient='no'}
%
% Density are expressed in: (as in Lisle & Leslon, 2004)
%
% (nb_of_point_in_sampling_windows / total_number_of_point)
%       divided by
% (area_of_sampling_windows / total_area)
%
% unit is % per %area or unitless
%
% Lisle, R. J. and Leyshon, P. R. 2004. Stereographic projection techniques
% for geologists and civil engineers. 2nd edition. Cambridge University
% Press. ISBN 0-521-53582-4
%
% See also stereoplot, stereocircle, stereoginput, stereogrid.
%
% B. Valley 08.08.2006
if nargin==1
    projection='equalangle';
    ccsize=1/50;
    meanorient='no';
elseif nargin==2
    ccsize=1/50;
    meanorient='no';
elseif nargin==3
    meanorient='no';
elseif nargin>4
    warning('to much input, extra input ignored')
end

if length(D(1,:))==2 % if weight not attributed, give weight 1 to all data
    D=[D ones(length(D(:,1)),1)];
end


[N c] = size(D);
wN=sum(D(:,3));
cD  = [cosd(90-D(:,2)).*sind(D(:,1)+180)...
    cosd(90-D(:,2)).*cosd(D(:,1)-180)...
    -sind(90-D(:,2))];      % direction cosines of data
cD=[cD; -cD]; % double data to cover both hemisphere (trick to avoid problems in stereoborder)
Dd=[D(:,3); D(:,3);];


gridstep=2;
rtheta=acosd(1-ccsize);

az = (0:gridstep:360)';   % grid in azim.
if az(end)~=360; az=[az; 360]; end
[Na,c] = size(az);
dip = (0:gridstep:90)';  % grid in dips
if dip(end)~=90; dip=[dip; 90]; end
[Nd,c] = size(dip);
Ng = Na*Nd;               % no of grid points
ki =(1:Na);
kj =(1:Nd);
i = reshape((ones(Nd,1)*ki)',1,Ng);
j = reshape(ones(Na,1)*kj,1,Ng);
Xg = [az(i) dip(j)];     % az and dip of all grid points
Dg  = [cosd(90-Xg(:,2)).*sind(Xg(:,1)+180)...
    cosd(90-Xg(:,2)).*cosd(Xg(:,1)+180)...
    -sind(90-Xg(:,2))];     % direction cosines of grid point

%R=acosd(cD*Dg'); % angles [°] between fract and grid points [nFrac x nGridpoint] this line produced a two heavy matrix and memory problem - the calcualation have been included in the here under loop in order to avoid this problem
Rcount=zeros(Ng,1); % counting for each point grid
for m=1:Ng;
    R=acosd(cD*Dg(m,:)');
    ix=find(R(:,1)<=rtheta);
    Rcount(m)=(sum(Dd(ix,1))/wN)/ccsize;
end
clear R;

if strcmp(projection,'equalangle') % stereographic projection
    r=tand((Xg(:,2))/2);
elseif strcmp(projection,'equalarea')
    r=sqrt(2)*cosd((180-Xg(:,2))/2);
end
x=sind(Xg(:,1)+180).*r;
y=cosd(Xg(:,1)+180).*r;
[t_b, m1, t_n] = unique([x y],'rows'); % delete repeted points
x=x(m1);
y=y(m1);
Rcount=Rcount(m1);

% for i=1:length(x)
%     text(x(i),y(i),sprintf('%1.0f',Rcount(i)))
% end

% grid creation
gs=0.005;
xg=-01:gs:1;
gs=gs + (1-xg(end))/(length(xg)-1); % adapt to go exactly from -1 to 1
xg=-1:gs:1;
[Xgrid,Ygrid] = meshgrid(xg,xg);

Zgrid = griddata(x,y,Rcount,Xgrid,Ygrid,'linear'); % interpolate on grid
Rd=sqrt(Xgrid.^2+Ygrid.^2);
Zgrid(find(Rd>1))=NaN;

[C,h]=contour(Xgrid,Ygrid,Zgrid,20);
hold on;
clear C;

% annotations
ht=text(0.8,0.8,sprintf('N=%1.0f',length(D(:,1))),...
    'FontSize',8);
if strcmp(meanorient,'yes')
    [ta,tp,tt]=meanorientation( D(:,[1 2]) , D(:,[3]) );
    text(0.8,-0.8,{'mean orientation :' sprintf('%1.1f / %1.1f \\pm%1.1f',ta,tp,tt)},...
        'FontSize',8);
end
if nargout==1;
    varargout{1}=h;
elseif nargout==2
    varargout{1}=h;
    varargout{2}=ht;
elseif nargout==4 & strcmp(meanorient,'yes');
    varargout{1}=h;
    varargout{2}=ta;
    varargout{3}=tp;
    varargout{4}=tt;
elseif nargout==5 & strcmp(meanorient,'yes');
    varargout{1}=h;
    varargout{1}=ht;
    varargout{2}=ta;
    varargout{3}=tp;
    varargout{4}=tt;
elseif nargout==0
else
    warning('Not an appropriate number of ouput arguments.')
end
return;