% Starter code prepared by James Hays

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
function [train_image_paths, test_image_paths, train_labels, test_labels] = ... 
    get_image_paths(data_path, categories, numTrainImagePerCate ,numTestImagePerCate)


% A = exist(numTestImagePerCate)


num_categories = length(categories); %number of scene categories.

totalTrainImage =sum(numTrainImagePerCate);

%This paths for each training and test image. By default it will have 1500
%entries (15 categories * 100 training and test examples each)
train_image_paths = cell(totalTrainImage, 1);
% test_image_paths  = cell(num_categories * num_train_per_cat, 1);
test_image_paths  =0;
%The name of the category for each training and test image. With the
%default setup, these arrays will actually be the same, but they are built
%independently for clarity and ease of modification.
train_labels = cell(totalTrainImage, 1);
% test_labels  = cell(num_categories * num_train_per_cat, 1);
test_labels  =0;



for i=1:num_categories
    
   imagesA = dir( fullfile(data_path , 'train' , categories{i},'face',   '*.jpg'));
   imagesB = dir( fullfile(data_path ,'train'  , categories{i},'face',   '*.JPG'));
   
   images = [imagesA;imagesB];

   for j=1 : numTrainImagePerCate(i)
       train_image_paths{  sum(numTrainImagePerCate( 1: (i-1)) ) + j } = fullfile(data_path, 'train', categories{i},'face', images(j).name);
       train_labels{  sum(numTrainImagePerCate( 1: (i-1)) ) + j } = categories{i};
   end   % (i-1)*numTrainImagePerCate(i-1)

   
%    images = dir( fullfile(data_path, 'test', categories{i}, '*.jpg'));
%    for j=1:num_train_per_cat
%        test_image_paths{(i-1)*num_train_per_cat + j} = fullfile(data_path, 'test', categories{i}, images(j).name);
%        test_labels{(i-1)*num_train_per_cat + j} = categories{i};
%    end
end


