figure; hold on; grid on;
for i = 1:size(method,2)
	mem = zeros(1,length(hbits));
    MAP = [];
	for j= 1:length(hbits)
		mem(j) = memory{j,i};
        MAP = [MAP, map{j, i}];
	end
	plot(MAP,mem,[color(i),'-o'],'linewidth',1);
	xlabel('MAP');
	ylabel('Memory Consumption');
	title(dataset);
	legend(method);
	box on; hold off;
end
