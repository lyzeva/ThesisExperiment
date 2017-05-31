line_width = 2;
marker_size = 8;
xy_font_size = 14;
legend_font_size = 9;
linewidth = 1.6;
title_font_size = xy_font_size;

for loop = 1:assess.loop
figure; hold on;
for i = size(assess.method,2)
    MAP = [];
    for j = 1: length(assess.hbits)
%         map{j,i} = area_RP(recall{j,i}, precision{j,i});
        MAP = [MAP, res.map{loop}{j, i}];
    end
    p = plot(log2(assess.hbits), MAP);
    marker=gen_marker(i);
    set(p,'Color', gen_color(i));
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end
end

h1 = xlabel('Number of bits');
h2 = ylabel('mean Average Precision (mAP)');
title(assess.dataset, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
% axis square;
set(gca, 'xtick', log2(assess.hbits));
set(gca, 'XtickLabel', {'16', '32', '64', '100'});
set(gca, 'linewidth', linewidth);
axis([log2(16),log2(100),0.35,0.86]);
hleg = legend(assess.method);
set(hleg, 'FontSize', legend_font_size);
set(hleg, 'Location', 'best');
box on; grid on; hold off;
end
