function [Fout] = Fresnel_propagation(Fin,lambda,dim,z)
%FRESNEL_PROPAGATION calculate the fresnel_propagation based on ASM
%   Detailed explanation goes here
[M,N] = size(Fin);
k = 2*pi/lambda;
deltax = dim/M;
deltay = dim/N;
x = -M/2:M/2-1;
y = -N/2:N/2-1;
Fx = 1/dim * x;
Fy = 1/dim * y;
[Fxx,Fyy] = meshgrid(Fx,Fy);
H = exp(1j*k*z*(1-(lambda*Fxx).^2-(lambda*Fyy).^2).^0.5);
A0 = fftshift(fft2(fftshift(Fin)));
U = A0.*H;
Fout = ifftshift(ifft2(ifftshift(U)));
end

