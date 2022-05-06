% Locust data preprocessing

load data_recording.mat

for f = 1:6
    seqnum = 1;
    dirname = strcat("sequence_", num2str(seqnum));
    mkdir(dirname);

    data = recording(1).data{seqnum,2};

    [numloc, frames] = size(data);
    for i = 1:frames
        frame = data(:,i);
        featarray = zeros(length(frame),10);
        count = 0;
        for j = 1:length(frame)
            if ~isempty(frame(j).features)
                count = count+1;
                featarray(count,1:9) = frame(j).features;
                featarray(count,10) = j;
            end
        end
        featarray = featarray(1:count,:);
        save(strcat(dirname,"/frame_",num2str(i),".mat"), "featarray");
    end
end