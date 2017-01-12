%% training images read
clear all;
clc;

data_path = '../data/';
categories = {'A-gun','A-Vai','Ding-Ding','Lally','Meatball','MuoMuo','O-tun','Soy-Milk', ...
    'Towel','Eric','Gouda','Hanamaki','Harry','Hei-lu-lu','Jackal','Jimmy','Peach','Pi-dan', ...
    'Sheriff','Tea-Tea','White','yoyo'};
numCate = length(categories);
numTrainImagePerCate = ones(numCate, 1);


for i = 1 : numCate
    images1 = dir( fullfile(data_path ,'train'  , categories{i}, 'face', '*.jpg'));
    %images2 = dir( fullfile(data_path ,'train'  , categories{i}, 'whole', '*.JPG'));
    %images3 = [images1;images2];
    numTrainImagePerCate(i) = size(images1 ,1);
    
end

numTestImagePerCate = 0;

 [train_image_paths, test_image_paths, train_labels, test_labels] = ...
     get_dataAug_image_paths(data_path, categories, numTrainImagePerCate ,numTestImagePerCate);


%% flipA-gun
for i = 1 : numCate
        for j = 1 : numTrainImagePerCate(i)
                
            im = imread(train_image_paths{sum(numTrainImagePerCate( 1: (i-1)) ) + j});
            imgMirror = flipdim(im, 2);
            path  = '../data/train/';
            index = num2str(j);
            path = [path   'aug/'  categories{i} '/flip' index '.jpg'];
            imwrite(imgMirror, path);
            
        end
end

%% color casting

for i = 1 : numCate
        for j = 1 : numTrainImagePerCate(i)
                
            im = imread(train_image_paths{sum(numTrainImagePerCate( 1: (i-1)) ) + j});
            imgColorR = im;
            imgColorG = im;
            imgColorB = im;
            [width height band] = size(im);
            for w = 1 : width
                for h = 1 : height
                    if(im(w, h, 1)*1.2 < 255)
                        imgColorR(w, h, 1) = im(w, h, 1)*1.2;
                    else
                        imgColorR(w, h, 1) = 255;
                    end
                    
                    if(im(w, h, 2)*1.2 < 255)
                        imgColorG(w, h, 2) = im(w, h, 2)*1.2;
                    else
                        imgColorR(w, h, 2) = 255;
                    end
                    
                    if(im(w, h, 3)*1.2 < 255)
                        imgColorR(w, h, 3) = im(w, h, 3)*1.2;
                    else
                        imgColorR(w, h, 3) = 255;
                    end
                    
                end
            end
            path  = '../data/train/';
            index = num2str(j);
%             [path   'aug/'  categories{i} '/flip' index '.jpg'];
%             pathR = [path  categories{i}  '/aug/colorR_'  index  '.jpg'];
            pathR =  [path   'aug/'  categories{i} '/colorR' index '.jpg'];
            imwrite(imgColorR, pathR);
%             pathG = [path  categories{i}  '/aug/colorG_'  index  '.jpg'];
%             imwrite(imgColorG, pathG);
%             pathB = [path  categories{i}  '/aug/colorB_'  index  '.jpg'];
%             imwrite(imgColorB, pathB);
            
        end
end


%% crop  x0.6 randomly

for i = 1 : numCate
        for j = 1 : numTrainImagePerCate(i)
                
            im = imread(train_image_paths{sum(numTrainImagePerCate( 1: (i-1)) ) + j});
            [width height band] = size(im);
            r = 0.6;
            for k = 1 : 3                
                xmin = unidrnd(floor(width*(1-r)));
                ymin = unidrnd(floor(height*(1-r)));
                region = [floor(r*width), floor(r*height)];
                rect = [xmin ,ymin, region];
                imgCrp = imcrop(im,rect);
                path  = '../train/';
                index = [num2str(j) '_' num2str(k)];
                path = [path  categories{i}  '/aug/crop_'  index  '.jpg'];
                imwrite(imgCrp, path);
            end
        end
end
