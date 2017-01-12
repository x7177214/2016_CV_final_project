%% testing and trainingimage read
clear
close
clc

run('vlfeat-0.9.20-bin/vlfeat-0.9.20/toolbox/vl_setup')
data_path = '../data/';
categories = {'A-gun','A-Vai','Ding-Ding','Lally','Meatball','MuoMuo','O-tun','Soy-Milk', ...
    'Towel','Eric','Gouda','Hanamaki','Harry','Hei-lu-lu','Jackal','Jimmy','Peach','Pi-dan', ...
    'Sheriff','Tea-Tea','White','yoyo'};
colorCate = {'black' , 'white' , 'brown'};

numCate = length(categories);

numTrainImagePerCate = ones (numCate, 1) ;


  % counts #of images per category
for i = 1 : numCate
 
    images1 = dir( fullfile(data_path ,'train'  , categories{i} , 'whole',   '*.jpg'));
    %images2 = dir( fullfile(data_path ,'train'  , categories{i}, 'face',   '*.JPG'));
    
    %images3 = [images1;images2];

    numTrainImagePerCate(i) = size(images1 ,1);

end
    numTestImagePerCate = 0;

  % get paths of training and testing images
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, numTrainImagePerCate ,numTestImagePerCate);

        
%%  color cluster
     
        color = cell(22,4);
        
        for i = 1 : 22
            color{i,3} = categories{i};
            color{i,2} = 'brown';
        end
        
        color{3,2}= 'black';
        color{4,2}= 'black';
        color{5,2}= 'black';
        color{7,2}= 'black';
        color{9,2}= 'black';
        color{12,2}= 'black';
        color{14,2}= 'black';
        color{17,2}= 'black';
        color{22,2}= 'black';
        
        color{21,2}= 'white';
        color{8,2}= 'white';

        load('allColor.mat');

        d =zeros(1,3);
        
        weight = ones (1,256);
        weight(1, 110 :155 ) = 1.7;
        weight(1, 235 :256 ) = 0.3;
        weight(1, 1:16) = 0.7;
        
        for i = 1:numCate
            
            counter = 0;
            
            for j = 1: numTrainImagePerCate(i)
               im = imread(train_image_paths{sum(numTrainImagePerCate( 1: (i-1)) ) + j});
               
               im = imresize(im,[100,100]);
               
               im = imcrop(im , [20 20 60 60]);
               
               im = rgb2gray(im);
               im = im(:);
               im = single(im);
               h = hist(im,256);
               h = h / sum(h);
               
 
               d(1) = sum (((h - black).*weight).^2 );%dblack
               d(2) = sum (((h - white).*weight).^2); %dwhite
               d(3) = sum (((h - brown).*weight).^2); %dBrown
               
               [~ , idx ] = min(d);
%                color{i,1} = colorCate{idx};
               
               counter  =counter + strcmp( colorCate{idx} ,color{i,2} );
            end
            color{i,4} = counter /numTrainImagePerCate(i) * 100;
        end
    
    %% HOG feature
 
    feature_params = struct('template_size', 36, 'hog_cell_size', 6);
    [~,~,~] = mkdir('visualizations');
    
    for i=1:numCate
        train_path = dir(fullfile(data_path,'train',categories{i},'whole','*.jpg'));
        num_images = length(train_path);
        HOG_size = zeros(1,(feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
        for j=1: num_images
            temp = fullfile(data_path,'train',categories{i},'whole', train_path(j).name);
            I = imread(temp);
            I = imresize(I,[36,36]);
            I = single(I)/255;
            I = rgb2gray(I);
            
            HOG = vl_hog(I,feature_params.hog_cell_size);
            HOG = reshape(HOG,size(HOG_size));
            feature(sum(numTrainImagePerCate( 1: (i-1)) ) + j,:) = HOG;
        end
    end
    
    %% White Train Classer
    
    lamda = 0.0001;
    j = 1;
    c = 1;
    for k = 1:numCate
        if color{k,2} == colorCate{2}
            d(j) = numTrainImagePerCate(k);
                  
            for i=1:d(j)
                x_white(:,i+sum(d(1:j-1))) = feature(sum(numTrainImagePerCate( 1: (k-1)) )+i,:);
                y_white(i+sum(d(1:j-1))) = c;
            end
            c = -1;
            j = j+1;
        end
    end
    [w_white,b_white] = vl_svmtrain(x_white, y_white, lamda);
    

    % Black Train Classer
    
    blackSVM = cell(9,2);
    for t = 1:9
        j = 1;
        y_black = -1*ones(1,650);
        for k = 1:numCate
            if color{k,2} == colorCate{1}
                d(j) = numTrainImagePerCate(k);

                for i=1:d(j)
                    x_black(:,i+sum(d(1:j-1))) = feature(sum(numTrainImagePerCate( 1: (k-1)) )+i,:);
                    if j==t
                        y_black(i+sum(d(1:j-1))) = 1;
                        blackSVM{t,3} = color{k,3};
                    end
                end
                j = j+1;
            end
        end
        [blackSVM{t,1},blackSVM{t,2}] = vl_svmtrain(x_black, y_black, lamda);  
    end
    
    
    % Brown Train Classer
    
      brownSVM = cell(11,2);
    for t = 1:11
        j = 1;
        y_brown = -1*ones(1,566);
        for k = 1:numCate
            if color{k,2} == colorCate{3}
                d(j) = numTrainImagePerCate(k);

                for i=1:d(j)
                    x_brown(:,i+sum(d(1:j-1))) = feature(sum(numTrainImagePerCate( 1: (k-1)) )+i,:);
                    if j==t
                        y_brown(i+sum(d(1:j-1))) = 1;
                        brownSVM{t,3} = color{k,3};
                    end
                end
                j = j+1;
            end
        end
        
        [brownSVM{t,1},brownSVM{t,2}] = vl_svmtrain(x_brown, y_brown, lamda);   
    end
    
    
    %% Test
    
    load('allColor.mat');
    for k = 1:numCate
        test_path = dir(fullfile(data_path,'test','whole',categories{k},'*.jpg'));
        num_images(k) = length(test_path);
        for j = 1:num_images
            temp = fullfile(data_path,'test','whole',categories{k},test_path(j).name);
            Cate{sum(num_images(1:(k-1)))+j,1} = categories{k};

            im = imread(temp);
            %im = flip(im,2);
            test_color = color_cluster(im, colorCate, black, white, brown);
            im = rgb2gray(im);
            im = single(im)/255;

            if test_color == colorCate{2}
                d = ((feature_params.template_size/feature_params.hog_cell_size)^2)*31;
                %test = flip(im,2);
                test = imrotate(im,5);
                test = imresize(test,[36,36]);
                test_HOG = vl_hog(test,feature_params.hog_cell_size);
                test_HOG = reshape(test_HOG,size(HOG_size));

                s = test_HOG*w_white+b_white;
                if s >=0
                    Cate{sum(num_images(1:(k-1)))+j,2} = categories{8};
                else
                    Cate{sum(num_images(1:(k-1)))+j,2} = categories{21};
                end
            end

            if test_color == colorCate{1}
                %test = flip(im,2);
                test = imrotate(im,5);
                test = imresize(test,[36,36]);
                test_HOG = vl_hog(test,feature_params.hog_cell_size);
                test_HOG = reshape(test_HOG,size(HOG_size));
                s = zeros(9,1);

                for i=1:9
                    s(i) = test_HOG*blackSVM{i,1}+blackSVM{i,2};
                end
                [~,par] = max(s);
                Cate{sum(num_images(1:(k-1)))+j,2} = blackSVM{par,3};

            end

            if test_color == colorCate{3}
                %test = flip(im,2);
                test = imrotate(im,5);
                test = imresize(test,[36,36]);
                test_HOG = vl_hog(test,feature_params.hog_cell_size);
                test_HOG = reshape(test_HOG,size(HOG_size));
                s = zeros(11,1);

                for i = 1:11
                    s(i) = test_HOG*brownSVM{i,1}+brownSVM{i,2};
                end
                [~,par] = max(s);
                Cate{sum(num_images(1:(k-1)))+j,2} = brownSVM{par,3};

            end

        end
    end
    acc = 0;
    for i = 1:220
        acc = acc+strcmp(Cate{i,1},Cate{i,2});
    end
    [num_test,~] = size(Cate);
    acc = acc/num_test;
    
    
    
    
 
