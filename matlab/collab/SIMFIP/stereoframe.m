function h=stereoframe
% plot le cercle
% h=stereoframe
% handle on lines and text objects 
t_a=0:360;
t_x=sind(t_a);
t_y=cosd(t_a);

h=nan(6,1);

h(1)=plot(t_x,t_y,'k');
hold on
axis square
axis equal
% clear t_*

% plot les ticks
h(2)=plot([1 1.1],[0 0],'k');
h(3)=plot([-1 -1.1],[0 0],'k');
h(4)=plot([0 0],[1 1.1],'k');
h(5)=plot([0 0],[-1 -1.1],'k');
set(gca,'visible','off')
set(gca,'Xlim',[-1.1 1.1])
set(gca,'Ylim',[-1.1 1.1])
set(gcf,'Color',[1 1 1])
h(6)=text(0.00,1.1,'N',...
    'FontSize',get(gca,'FontSize'),...
    'HorizontalAlignment','Center',...
    'VerticalAlignment','Bottom');