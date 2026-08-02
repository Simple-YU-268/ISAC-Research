function draw_system_architecture(output_dir)
%DRAW_SYSTEM_ARCHITECTURE  Figure 1: Cell-Free ISAC signal architecture.

if nargin < 1 || isempty(output_dir)
    output_dir=fullfile(pwd,'..','..','experiment_packages','v1.0','figures');
end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
fig=figure('Visible','off','Position',[100 100 1200 680],'Color','w');
ax=axes(fig,'Position',[.04 .07 .92 .87]); hold(ax,'on'); axis(ax,[0 1 0 1]); axis(ax,'off');

% Central processor.
rectangle(ax,'Position',[.40 .82 .20 .10],'Curvature',.08,'FaceColor',[.92 .92 .92], ...
    'EdgeColor',[.15 .15 .15],'LineWidth',1.4);
text(ax,.50,.87,'Central processor','HorizontalAlignment','center','FontWeight','bold');
text(ax,.50,.835,'Joint beamforming and sensing clustering','HorizontalAlignment','center','FontSize',9);

% AP positions and CPU fronthaul links.
ap=[.12 .62;.28 .62;.44 .62;.60 .62;.76 .62;.90 .62];
for m=1:6
    plot(ax,[.50 ap(m,1)],[.82 ap(m,2)+.04],'--','Color',[.25 .25 .25],'LineWidth',.9);
    rectangle(ax,'Position',[ap(m,1)-.045 ap(m,2)-.035 .09 .07],'Curvature',.08, ...
        'FaceColor',[.80 .84 .90],'EdgeColor',[.10 .20 .40],'LineWidth',1.2);
    text(ax,ap(m,1),ap(m,2),sprintf('AP %d',m),'HorizontalAlignment','center','FontWeight','bold');
end
text(ax,.03,.73,'Fronthaul','FontSize',9,'Color',[.2 .2 .2]);

% UEs and targets.
ue=[.17 .20;.38 .20;.59 .20]; target=[.76 .22;.91 .22];
for k=1:3
    scatter(ax,ue(k,1),ue(k,2),150,[.08 .55 .25],'d','filled','MarkerEdgeColor','k');
    text(ax,ue(k,1),ue(k,2)-.055,sprintf('UE %d',k),'HorizontalAlignment','center','FontWeight','bold');
end
for p=1:2
    scatter(ax,target(p,1),target(p,2),240,[.85 .12 .12],'p','filled','MarkerEdgeColor','k');
    text(ax,target(p,1),target(p,2)-.06,sprintf('Target %d',p),'HorizontalAlignment','center','FontWeight','bold');
end

% Global cooperative communication links (blue).
for m=1:6
    for k=1:3
        plot(ax,[ap(m,1) ue(k,1)],[ap(m,2)-.04 ue(k,2)+.035],'-', ...
            'Color',[.15 .38 .82 .22],'LineWidth',.9,'HandleVisibility','off');
    end
end
% Dedicated target-specific sensing links (red); illustrative b=1 entries.
active={ [1 3 5], [2 4 6] };
for p=1:2
    for m=active{p}
        plot(ax,[ap(m,1) target(p,1)],[ap(m,2)-.04 target(p,2)+.045],'-', ...
            'Color',[.88 .20 .08 .78],'LineWidth',2.0,'HandleVisibility','off');
    end
end

% Legend-like explanatory keys.
plot(ax,[.06 .16],[.43 .43],'-','Color',[.15 .38 .82],'LineWidth',2);
text(ax,.18,.43,'Global cooperative communication beams W_k (not gated by b_mp)', ...
    'VerticalAlignment','middle','FontSize',10,'Interpreter','none');
plot(ax,[.06 .16],[.37 .37],'-','Color',[.88 .20 .08],'LineWidth',2.5);
text(ax,.18,.37,'Dedicated sensing waveform S_p, enabled only when b_mp = 1', ...
    'VerticalAlignment','middle','FontSize',10,'Interpreter','none');
text(ax,.50,.98,'Cell-Free ISAC architecture with asymmetric sensing gating', ...
    'HorizontalAlignment','center','FontWeight','bold','FontSize',14);
text(ax,.50,.02,'b_mp authorizes AP m to transmit the target-p dedicated sensing waveform; communication remains globally cooperative.', ...
    'HorizontalAlignment','center','FontSize',10,'Interpreter','none');
exportgraphics(fig,fullfile(output_dir,'fig1_system_architecture.png'),'Resolution',300);
close(fig);
end
