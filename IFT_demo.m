%pkg load image; % <-- Comment this for MATLAB
addpath('C:\Users\felip\Desktop\Projetos\main-ms-felipe\src\main\java\com\rodolfo\ulcer\segmentation\repositories\ift\mex');

img = imread("man.png");
img = int32(img);

[label_img, border_img] = IFT_Superpixels(img, 8000, 50);

subplot(1,2,1),imshow(uint8(label_img)),...
subplot(1,2,2),imshow(logical(border_img));
