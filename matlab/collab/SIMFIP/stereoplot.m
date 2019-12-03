function [y]=stereoplot(varargin)
% STEROPLOT
% [y]=stereoplot(data,projection,symbol,color,high)
% INPUT
% data:
%   azimuth-pendage de plans
% projection:
%   'equalangle'
%   'equalarea'
% symbol:
%   standard matlab plotting symbol (see plot)
%   si pas spécifié, '.b'
% color:
%   [rvb]
% high:
%   z value of the plot [default = 0]
% OUTPUT
% y:
%   handle sur le lineserie

% manage les argument d'entrée
if (nargin<2)
    disp('ERROR: not enough input arguments!');
    return;
end % if

switch nargin
    case 2
        data  = varargin{1};
        projection = varargin{2};
        symbol='.b';
        high=0;
    case 3
        data  = varargin{1};
        projection = varargin{2};
        symbol= varargin{3};
        high=0;
    case 4
        data  = varargin{1};
        projection = varargin{2};
        symbol= varargin{3};
        color=varargin{4};
        high=0;
    case 5
        data  = varargin{1};
        projection = varargin{2};
        symbol= varargin{3};
        color=varargin{4};
        high=varargin{5};
end

% % plot le cercle
% t_a=[0:360];
% t_x=sind(t_a);
% t_y=cosd(t_a);
% t_z=(t_x*0+1)*high;
% 
% plot3(t_x,t_y,t_z,'k');
axes=(gca);
hold on
axis square
axis equal
% clear t_*

% % plot les ticks
% plot3([1 1.1],[0 0],[high high],'k');
% plot3([-1 -1.1],[0 0],[high high],'k');
% plot3([0 0],[1 1.1],[high high],'k');
% plot3([0 0],[-1 -1.1],[high high],'k');


% prépare les données
if length(data)==0
    set(gca,'visible','off')
    set(gca,'Xlim',[-1.1 1.1])
    set(gca,'Ylim',[-1.1 1.1])
    set(gcf,'Color',[1 1 1])
    % text(-0.05,1.2,high,'N');
    view([0 90]);
    y=[];
    return;
end
poles=data;
poles(:,2)=90-data(:,2);
poles(:,1)=data(:,1)+180;
poles(:,1)=poles(:,1)-(poles(:,1)>=360)*360;
switch lower(projection)
    case 'equalangle'
        t_r=tand((90-poles(:,2))/2);
    case 'equalarea'
        t_r=sqrt(2)*cosd((90+poles(:,2))/2);
end
t_x=sind(poles(:,1)).*t_r;
t_y=cosd(poles(:,1)).*t_r;
t_z=(t_x.*0+1).*high;
if nargin<4;
    y=plot3(t_x,t_y,t_z,symbol);
else
    y=plot3(t_x,t_y,t_z,symbol,'Color',color);
end 
set(gca,'visible','off')
set(gca,'Xlim',[-1.1 1.1])
set(gca,'Ylim',[-1.1 1.1])
set(gcf,'Color',[1 1 1])
%text(-0.05,1.2,high,'N');
% text(0.8,0.8,sprintf('N=%1.0f',length(data(:,1))),...
%      'FontSize',8);
if nargin<5
    view([0 90]);
end

