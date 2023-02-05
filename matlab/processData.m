%%
close all; clear; clc;
cd

%%
img = im2double(im2gray(imread("Data/offaxis/holo1.bmp")));
figure;imagesc(img);title('CapturedHologram');axis image;
img=imcrop(img,[0,0,1024,1024]);
[nx, ny]=size(img);

% padding = false;
% if padding
%     I=zeros(5*nx);
%     I(2*nx:3*nx,2*nx:3*nx)=img;
% else
%     I=img;
% end
% 
% A0=fftshift(fft2(fftshift(I)));
% imshow(abs(A0));title("Spectrum of hologram")

%% Reconstruction
hologram = img;
% spectrum = FT2Dc(hologram);
spectrum=fftshift(fft2(fftshift(hologram)));
spectrum_abs = abs(spectrum);
figure, imshow(log(spectrum_abs), []);
% figure, imshow(flipud(rot90(log(spectrum_abs))), []);
axis on
set(gca,'YDir','normal')
% colormap('gray')
colorbar;   
%%
lambda = 650*10^-9;
pixel_size = 3.45*10^-6;
z = 15.5*10^-2;

% Blocking the central part of the spectrum
R0 = 50;
spectrum_abs1 = zeros(nx,ny); 
for ii=1:nx
    for jj=1:ny
     
    x = ii - nx/2;
    y = jj - ny/2;
    
    if (sqrt(x^2 + y^2) > R0) 
        spectrum_abs1(ii, jj) = spectrum_abs(ii,jj); 
    end
    end
end
% Blocking half of the spectrum
spectrum_abs1(1:nx/2,:) = 0;
% Blocking 1/4 of the spectrum
spectrum_abs1(nx/2:nx,ny/2:ny) = 0;
% imshow(flipud(rot90(log(spectrum_abs1))), []);
figure, imshow(log(spectrum_abs1), []);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the position of the side-band in the spectrum
maximum = max(max(spectrum_abs1));
[x0, y0] = find(spectrum_abs1==maximum)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Shifting the complex-valued spectrum to the center
% N=nx;
% spectrum2 = zeros(N,N);
% 
% if (N+1)//2 
% 
% 
% x0 = x0 - N/2 - 1;
% y0 = y0 - N/2 - 1;
% 
%     for k, extra_shift in zip(axes, additional_shift):
%         n = tmp.shape[k]
%         if (n+1)//2 - extra_shift < n:
%             p2 = (n+1)//2 - extra_shift
%         else:
%             p2 = abs(extra_shift) - (n+1)//2
%         mylist = np.concatenate((np.arange(p2, n), np.arange(0, p2)))
%         y = np.take(y, mylist, k)
% 
% 
% for ii = 1:N-x0
%     for jj = 1:N-y0    
%         spectrum2(ii, jj) = spectrum(ii+x0,jj+y0); 
%     end
% end
