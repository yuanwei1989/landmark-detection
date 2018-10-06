% Builds a statistical shape model of the 10 landmarks of the fetal head.
%
% This Script is adapted from:
% https://uk.mathworks.com/matlabcentral/fileexchange/26706-active-shape-model-asm-and-active-appearance-model-aam
%
% Literature used: Ginneken B. et al. "Active Shape Model Segmentation 
% with Optimal Features", IEEE Transactions on Medical Imaging 2002.
%
% Functions are written by D.Kroon University of Twente (February 2010)
%
% Modified by Yuanwei Li (2018)

clear all; clc; close all;
addpath('./Functions/')
addpath('./NIfTI_reader/')


%% Set options
% folder containing the training landmarks
landmarkFolder='/vol/medic01/users/yl5709/iFIND/Data/Feta_Head_30_149/Isovoxel_Align/Fixed_Size_324_207_279/Landmarks/';
% file containing list of shape images to use
shapeListFile = '/vol/medic01/users/yl5709/iFIND/Data/Feta_Head_30_149/File_List/list_all_train.txt';
% folder storing the shape model txt and mat file
shapeModelFolder='./shape_model/';
% Percentage of variance used to keep the eigenvectors used in the model. (ie. 0.98)
options.eigVecPer=1;
% New image size that is required by the CNN 
imgSizeCNN = [324 207 279];
% unwanted landmark indices for aligned images
landmark_unwant = [1 9 10 14 15 16];
% If verbose is true all debug images will be shown.
options.verbose=true;


if ~exist(shapeModelFolder, 'dir')
  mkdir(shapeModelFolder)
end
%% Load training data
% First Load the Landmarks Training DataSets
fileID = fopen(shapeListFile,'r'); 
ids = textscan(fileID,'%s');
ids = ids{1};
fclose(fileID);

TrainingData=struct;
num_ex = length(ids);
for i=1:num_ex
    disp(['loading image ' num2str(i) '/' num2str(num_ex)])
    
    % Load landmarks
    fid = fopen([landmarkFolder ids{i} '_ps.txt'], 'r');
    landmarks = fscanf(fid, '%f %f %f', [3 Inf]);
    landmarks = landmarks';
    fclose(fid);
    
    % Remove unwanted landmarks
    landmarks(landmark_unwant, :) = [];
    TrainingData(i).Vertices=landmarks;
end

num_landmarks = size(TrainingData(1).Vertices,1);

%% Shape Model %%
% Make the Shape model, which finds the variations between contours
% in the training data sets. And makes a PCA model describing normal
% contours
[ShapeData TrainingData]= MakeShapeModel(TrainingData,options.eigVecPer);

% Show +-3s.d. of each mode for the top six modes
w=1;
xrange=[1, imgSizeCNN(1)];
yrange=[1, imgSizeCNN(2)];
zrange=[1, imgSizeCNN(3)];
if(options.verbose)
    for i=1:min(6,length(ShapeData.Evalues))
        j=mod(i,2);
        if(j)
            h(w)=figure;
            w=w+1;
            j=1;
        else
            j=4;
        end
        subplot(2,3,j);
        xtest = ShapeData.x_mean - ShapeData.Evectors(:,i)*sqrt(ShapeData.Evalues(i))*3;
        xtest = (reshape(xtest, 3, num_landmarks))';
        scatter3(xtest(:,1), xtest(:,2), xtest(:,3), 36, (1:size(landmarks,1))', 'x');
        axis equal; xlabel('x'); ylabel('y'); zlabel('z');
        xlim(xrange); ylim(yrange); zlim(zrange);
        title(['b' num2str(i) '-3*sqrt(lambda' num2str(i) ')']);
        subplot(2,3,j+1);
        xtest = (reshape(ShapeData.x_mean, 3, num_landmarks))';
        scatter3(xtest(:,1), xtest(:,2), xtest(:,3), 36, (1:size(landmarks,1))', 'x');
        axis equal; xlabel('x'); ylabel('y'); zlabel('z');
        xlim(xrange); ylim(yrange); zlim(zrange);
        title(['b' num2str(i)]);
        subplot(2,3,j+2);
        xtest = ShapeData.x_mean + ShapeData.Evectors(:,i)*sqrt(ShapeData.Evalues(i))*3;
        xtest = (reshape(xtest, 3, num_landmarks))';
        scatter3(xtest(:,1), xtest(:,2), xtest(:,3), 36, (1:size(landmarks,1))', 'x');
        axis equal; xlabel('x'); ylabel('y'); zlabel('z');
        xlim(xrange); ylim(yrange); zlim(zrange);
        title(['b' num2str(i) '+3*sqrt(lambda' num2str(i) ')']);
    end
    drawnow;
    for i=1:w-1
        saveas(h(i),[shapeModelFolder 'Variation' num2str(i) '.fig']);
    end
end

% Save shape model in .mat file
ShapeData.landmark_unwant = landmark_unwant;
ShapeData.imgSizeCNN = imgSizeCNN;
save([shapeModelFolder 'ShapeModel.mat'],'ShapeData');
