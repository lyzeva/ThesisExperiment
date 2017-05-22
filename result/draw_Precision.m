line_width = 1.5;
marker_size = 7;
xy_font_size = 14;
legend_font_size = 12;
title_font_size = xy_font_size;
for j=1:length(hbits)
    figure;hold on;grid on;
    title([dataset]);
    bit = hbits(j)*hbits(j);
    for i= [1:7]
        p = plot(thresh_kNN{j,i}(1:2:end)*5, precision_kNN{j,i}(1:2:end)); 
        set(p,'Color', gen_color(i));
        set(p,'Marker', gen_marker(i));
        set(p,'LineWidth', line_width);
        set(p,'MarkerSize', marker_size);
    end
    xlabel('Number of Retrieved Points');
    ylabel(['Recall','@',num2str(bit),'bits']);
    legend(method{[1:7]});
    box on;hold off;
end