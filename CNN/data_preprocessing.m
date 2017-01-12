%% testing and training image read
clear
close
clc

% for training and testing on face
    data_path = '../data/face_DataAug';
    %data_path = '../data/face_forTest';

% for training and testing on whole body
    %data_path = '../data/whole_DataAug';
    %data_path = '../data/whole_forTest';


categories = {'A-gun','A-Vai','Ding-Ding','Lally','Meatball','MuoMuo','O-tun','Soy-Milk', ...
    'Towel','Eric','Gouda','Hanamaki','Harry','Hei-lu-lu','Jackal','Jimmy','Peach','Pi-dan', ...
    'Sheriff','Tea-Tea','White','yoyo'};


numCate = length(categories);

numTrainImagePerCate = ones(numCate, 1) ;


% counts #of images per category
for i = 1 : numCate
 
    images1 = dir( fullfile(data_path   , categories{i} ,   '*.jpg'));
    numTrainImagePerCate(i) = size(images1 ,1);

end
    numTestImagePerCate = 0;

  % get paths of training and testing images
[train_image_paths, test_image_paths,~, ~] = ...
    get_image_paths(data_path, categories, numTrainImagePerCate ,numTestImagePerCate);

totalTrain = sum(numTrainImagePerCate);

%% data storing
    
    Size = 64 ; %the image size fed into CNN
    
    train = [];
    trainLabel = zeros(totalTrain,  numCate);
    h = 1 ;

    
    for i = 1 : numCate
        for j  =  1 :  numTrainImagePerCate(i) 
                
                trainLabel(h,i) = 1;  %for label
                h = h+1;
            
                im = imread(train_image_paths{sum(numTrainImagePerCate( 1: (i-1)) ) + j});
                im = imresize(im , [Size ,Size]);
                im = im2double(im);
                
                [~,~,D] =size(im);
                
                
                % data encoding %
                GG = ones(Size^2,D);
                for k =1 : D
                    tmp = im(:,:,k)';
                    tmp = tmp(:);

                    GG(:,k) = tmp ;
                end
                GG = GG';
                unrollGG = GG(:); 
                unrollGG  =unrollGG' ;
                % data encoding end%
                
                train = [train ; unrollGG ];          
        end
    end

    
    % shuffle the order 
    ShuffleIdx = randperm(totalTrain);
    
    train = train(ShuffleIdx, :) ;
    trainLabel =trainLabel(ShuffleIdx, :);
    
    
    % Then please manually save the variables 'train' and 'trainLabel' as one .mat file

    % To save the testing data, change the variable names as 'test' and 'testLabel' 
