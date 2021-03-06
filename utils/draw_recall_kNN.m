line_width = 1.5;
marker_size = 7;
xy_font_size = 14;
legend_font_size = 12;
title_font_size = xy_font_size;
for j=1:length(assess.hbits)
    figure;hold on;grid on;
    title([dataset]);
    bit = assess.hbits(j)*assess.hbits(j);
    for i= [1:7]
        p = plot(thresh_kNN{j,i}, recall_kNN{j,i}); 
        set(p,'Color', gen_color(i));
        set(p,'Marker', gen_marker(i));
        set(p,'LineWidth', line_width);
        set(p,'MarkerSize', marker_size);
    end
    xlabel('Number of Retrieved Points');
    ylabel(['Recall','@',num2str(bit),'bits']);
    legend(assess.method{[1:7]});
    box on;hold off;
end