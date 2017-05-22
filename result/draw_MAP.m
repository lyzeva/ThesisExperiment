line_width = 2;
marker_size = 8;
xy_font_size = 14;
legend_font_size = 12;
linewidth = 1.6;
title_font_size = xy_font_size;

figure; hold on;
for i = 1:size(method,2)
    MAP = [];
    for j = 1: length(hbits)
%         map{j,i} = area_RP(recall{j,i}, precision{j,i});
        MAP = [MAP, map{j, i}];
    end
    p = plot(log2(hbits.*hbits), MAP);
    marker=gen_marker(i);
    set(p,'Color', gen_color(i));
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

h1 = xlabel('Number of bits');
h2 = ylabel('mean Average Precision (mAP)');
title(dataset, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
% axis square;
set(gca, 'xtick', log2(hbits.*hbits));
set(gca, 'XtickLabel', {'16', '64', '256', '1024'});
set(gca, 'linewidth', linewidth);
hleg = legend(method);
set(hleg, 'FontSize', legend_font_size);
set(hleg, 'Location', 'best');
box on; grid on; hold off;