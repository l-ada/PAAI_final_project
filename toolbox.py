import matplotlib.pyplot as plt
import numpy as np
import sys

import torch as tch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from torch.nn import Module
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop, pad


from math import pi, factorial

from IPython.display import clear_output


# Hout​=⌊stride×Hin​+2×padding−dilation×(kernel_size−1)−1​+1⌋
encoder_net_ae = tch.nn.Sequential(
    tch.nn.Conv2d(1,16,3,padding=1), 
    tch.nn.ReLU(),
    tch.nn.Conv2d(16,32,3,stride=2,padding=1), # 16x16
    tch.nn.ReLU(), 
    tch.nn.Conv2d(32,64,3,stride=2,padding=1), # 8x8
    tch.nn.ReLU(), 
    tch.nn.Conv2d(64,64,3,stride=2,padding=1), # 4x4
    tch.nn.ReLU(), 
    tch.nn.Conv2d(64,64,3,stride=2,padding=1), # 2x2
    tch.nn.ReLU(), 
    tch.nn.Flatten(), 
    tch.nn.Linear(256,128),
    tch.nn.ReLU(), 
    tch.nn.Linear(128,128))

# Hout​=(Hin−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
decoder_net_ae = tch.nn.Sequential(
    tch.nn.Linear(128,128),
    tch.nn.ReLU(),
    tch.nn.Linear(128,256), 
    tch.nn.Unflatten(1,unflattened_size=(64,2,2)), # 2x2
    tch.nn.ReLU(),
    tch.nn.ConvTranspose2d(64,64,3,
                            stride=2, 
                            padding=1,
                            output_padding=1),     # 4x4
    tch.nn.ReLU(), 
    tch.nn.ConvTranspose2d(64,64,3,
                            stride=2,
                            padding=1,
                            output_padding=1),     # 8x8
    tch.nn.ReLU(), 
    tch.nn.ConvTranspose2d(64,32,3,
                            stride=2,
                            padding=1,
                            output_padding=1),     # 16x16
    tch.nn.ReLU(), 
    tch.nn.ConvTranspose2d(32,16,3,
                            stride=2,
                            padding=1,
                            output_padding=1),     # 32x32
    tch.nn.ReLU(), 
    tch.nn.Conv2d(16,1,3,padding=1))

def fourier_whitening(img):
    n_patches, _, n_y, n_x = img.shape
    img_whittened = tch.fft.fft2(img.reshape(n_patches,1,n_y,n_x)) 
    img_power_spectrum = tch.sqrt((tch.abs(img_whittened)**2).mean(0)) 
    img_whittened /= img_power_spectrum
    img_whittened = tch.fft.ifft2(img_whittened).real
    # on renormalise
    img_whittened -= img_whittened.mean(0);
    img_whittened /= img_whittened.std(0)
    return img_whittened, img_power_spectrum

# convert rgb color image to gray level image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# display image
def disp(im_list, shape=(1,1), scale=1):
    fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1], squeeze=False)
    fig.set_size_inches(5*scale*shape[1]/2.54,5*scale*shape[0]/2.54)
    
    if shape[0]==shape[1]==1:
        im_list = [im_list]
    if len(im_list)>shape[0]*shape[1]:
        raise ValueError('The product of figure shape must be'+
                         ' lower than im_list length')
    
    k = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax[i,j].imshow(im_list[k], cmap='gray');
            ax[i,j].xaxis.set_visible(False)
            ax[i,j].yaxis.set_visible(False)
            k += 1
    return fig, ax


def disp_losses(total_loss, partial_losses, epoch):
    clear_output(wait=True)
    print('Epoch: %i -- Total loss: %.5f'%(epoch, total_loss[epoch].mean()))
    print('          -- Reconstruction loss: %.5f  (%.5f)'\
          %(partial_losses[epoch,:,0].mean(), partial_losses[epoch,:,0].mean()))
    print('          -- Sparsity loss: %.5f  (%.5f)'\
          %(partial_losses[epoch,:,1].mean(), partial_losses[epoch,:,1].mean()))
    sys.stdout.flush()  

def disp_loss(total_loss, epoch):
    clear_output(wait=True)
    print('Epoch: %i -- Total loss: %.5f'%(epoch, total_loss)) 

def disp_2losses(running_loss, epoch_loss, epoch):
    clear_output(wait=True)
    print('Epoch: %i -- Running loss: %.5f'%(epoch, running_loss))
    print('          -- Epoch loss: %.5f'%(epoch_loss))


# transform for varying size images
def loader(path):
    return read_image(path)

class CenterCropPad(tch.nn.Module):
    def __init__(self, n_i, n_j) -> None:
        super(CenterCropPad,self).__init__()
        self.n_i = n_i
        self.n_j = n_j

    def forward(self, input):
        _, n_y, n_x = input.shape
        if n_x>self.n_j and n_y>self.n_i:
            output = center_crop(input, (self.n_j, self.n_i))
        elif n_x>=self.n_j and n_y<=self.n_i:
            output = center_crop(input, (n_y,self.n_j))
            if (self.n_i-n_y)%2==0:
                pad_y_t = (self.n_i-n_y)//2
                pad_y_b = (self.n_i-n_y)//2
            elif (self.n_i-n_y)%2==1:
                pad_y_t = (self.n_i-n_y)//2
                pad_y_b = (self.n_i-n_y)//2+1
            output = pad(output,(0,pad_y_t,0,pad_y_b))
        elif n_x<=self.n_j and n_y>=self.n_i:
            output = center_crop(input, (self.n_i,n_x))
            if (self.n_j-n_x)%2==0:
                pad_x_l = (self.n_j-n_x)//2
                pad_x_r = (self.n_j-n_x)//2
            elif (self.n_j-n_x)%2==1:
                pad_x_l = (self.n_j-n_x)//2
                pad_x_r = (self.n_j-n_x)//2+1
            output = pad(output,(pad_x_l,0,pad_x_r,0))
        elif n_x<self.n_j and n_y<self.n_i:
            if (self.n_j-n_x)%2==0:
                pad_x_l = (self.n_j-n_x)//2
                pad_x_r = (self.n_j-n_x)//2
            elif (self.n_j-n_x)%2==1:
                pad_x_l = (self.n_j-n_x)//2
                pad_x_r = (self.n_j-n_x)//2+1
            if (self.n_i-n_y)%2==0:
                pad_y_t = (self.n_i-n_y)//2
                pad_y_b = (self.n_i-n_y)//2
            elif (self.n_i-n_y)%2==1:
                pad_y_t = (self.n_i-n_y)//2
                pad_y_b = (self.n_i-n_y)//2+1

            output = pad(input,(pad_x_l,pad_y_t,pad_x_r,pad_y_b))
        else:
            output = input

        return output.to(tch.float32)


# steerable pyramid

def low_filter(r):
    return ((r<=pi/4) + (r>pi/4)*(r<pi/2)*tch.cos(pi/2*(tch.log(4*r/pi)\
            /tch.log(tch.tensor(2))))).type(tch.complex64)

def high_filter(r):
    return ((r>=pi/2) + (r>pi/4)*(r<pi/2)*tch.cos(pi/2*(tch.log(2*r/pi)\
            /tch.log(tch.tensor(2))))).type(tch.complex64)

def steer_filter(t,k,n):
    alpha = 2**(n-1)*factorial(n-1)/(n*factorial(2*(n-1)))**(0.5)
    return ((alpha*tch.cos(t-pi*k/n)**(n-1))*(tch.abs(\
            tch.remainder(t+pi-pi*k/n,2*pi)-pi)<pi/2)).type(tch.complex64)


class LayerSteerablePyramid(Module):
    ''' 
    Compute the steerable decomposition at the first scale.
    '''
    def __init__(self, n_i = 256, n_j = 256, scale = 0, n_ori = 4,
                 is_first = False, real = False, up_sampled = False,
                 fourier = False, double_phase=False, device='cpu') -> None:
        super(LayerSteerablePyramid, self).__init__()
        self.n_ori = n_ori
        self.n_i = n_i
        self.n_j = n_j
        self.scale = scale
        self.n_i_ = n_i//2**scale
        self.n_j_ = n_j//2**scale
        self.is_first = is_first
        self.real = real
        self.up_sampled = up_sampled
        self.fourier = fourier
        self.double_phase = double_phase
        self.device = device

        if real == True and fourier == True:
            raise(ValueError('real and fourier cannot be simultaneously true'))
        
        l_y = tch.linspace(-self.n_i_//2,self.n_i_//2-1,self.n_i_,
                           device=self.device)
        l_x = tch.linspace(-self.n_j_//2,self.n_j_//2-1,self.n_j_,
                           device=self.device)
        
        self.x, self.y = tch.meshgrid(l_x, l_y, indexing='xy')
        r = tch.sqrt((self.x*2*pi/self.n_j_)**2\
                    +(self.y*2*pi/self.n_i_)**2)\
                        .view((1, 1, self.n_i_, self.n_j_))
        th = tch.atan2(self.y, self.x).view((1, 1, self.n_i_, self.n_j_))
        r[0, 0, self.n_i_//2, self.n_j_//2] = 1e-15

        if is_first==True:
            self.filter_l0 = low_filter(r/2)
            self.filter_h0 = high_filter(r/2)
            
        self.filter_l = low_filter(r)
        self.filter_h = high_filter(r)
        self.filter_ori = [steer_filter(th,k,n_ori) for k in range(n_ori)]
        self.filter_ori_sym = [steer_filter(th+pi,k,n_ori) 
                                    for k in range(n_ori)]
        
        self.zero_pad = None
        self.output = None
        self.low = None
        self.reconstructed_image = None

    def forward(self, input):

        # add a test to avoid up_sampled and self.is_first to be 
        # simultaneously true

        n_b, n_c = input.shape[:2]
        self.output = tch.zeros((n_b, n_c, self.n_ori+1+self.is_first*1,
                                 self.n_i_, self.n_j_), dtype=tch.complex64,
                                device=self.device)

        if self.is_first==True:
            self.output[:,:,0] = self.filter_h0*input
            low = self.filter_l0*input
        else: 
            low = input
            
        for k in range(self.n_ori):
            self.output[:,:,1*self.is_first+k] = \
                2.0*self.filter_ori[k]*self.filter_h*low
        
        self.low = self.filter_l*low
        self.output[:,:,-1] = 1.0*self.low
        
        # 
        if self.up_sampled == True and self.is_first == False:
            pad_i = ((self.n_i-self.n_i_)//2)
            pad_j = ((self.n_j-self.n_j_)//2)
            self.zero_pad = tch.nn.ZeroPad2d((pad_j,pad_j,pad_i,pad_i))
            self.output = self.zero_pad(self.output)

        if self.fourier != True:
            self.output = ifft2(ifftshift(self.output, dim=(-2,-1)))
            ### double the phase
            if self.double_phase == True and self.up_sampled == True\
                  and self.is_first == False:
                r = tch.abs(self.output)
                theta = tch.angle(self.output)
                self.output = r*tch.exp(2**self.scale*1j*theta)

        if self.real == True:
            self.output = self.output.real

        return self.output

    def recompose(self):
        if self.fourier != True and self.real != True:
            sub_bands = fftshift(fft2(self.output.real), dim=(-2,-1))
        elif self.fourier != True and self.real == True:
            sub_bands = fftshift(fft2(self.output), dim=(-2,-1))
        else:
            sub_bands = ifft2(ifftshift(self.output, dim=(-2,-1)))
            sub_bands = fftshift(fft2(sub_bands.real), dim=(-2,-1))

        if self.up_sampled == True and self.is_first == False:
            sub_bands = sub_bands[:,:,:,self.y.type(tch.long)+self.n_i//2,
                                        self.x.type(tch.long)+self.n_j//2]
            
        self.reconstructed_image = self.filter_l*sub_bands[:,:,-1]
        
        for k in range(self.n_ori):
            self.reconstructed_image += (self.filter_ori[k]
                                         +self.filter_ori_sym[k])\
                                        *self.filter_h\
                                        *sub_bands[:,:,1*self.is_first+k]

        if self.is_first == True:
            self.reconstructed_image *= self.filter_l0
            self.reconstructed_image += self.filter_h0*sub_bands[:,:,0]

        if self.up_sampled == True and self.is_first == False:
            self.reconstructed_image = self.zero_pad(self.reconstructed_image)

        return self.reconstructed_image
        

class SteerablePyramid(Module):
    ''' 
    Compute the Steerable Pyramid decomposition.

    Parameters 
    ----------
    n_i : int, default = 256
        image height 

    n_j : int, default = 256
        image width

    n_scale : int, default = 4
        number of scales of pyramid decomposition

    n_ori : int, default = 4
        number of orientation at each scale

    real : bool, default = False
        return only the real part of the steerable pyramid coefficients

    up_sampled : bool, default = False
        up sample the image sub-bands so that they have the size of the input
        image

    fourier : bool, default = False
        if True, return the Fourier transform of the wavelet sub-bands
    '''
    def __init__(self, n_i = 256, n_j = 256, n_scale = 4, n_ori = 4,
                 real = False, up_sampled = False, fourier = False,
                 double_phase=False, device='cpu') -> None:
        super(SteerablePyramid, self).__init__()
        self.n_ori = n_ori
        self.n_scale = n_scale
        self.n_i = n_i
        self.n_j = n_j
        self.real = real
        self.up_sampled = up_sampled
        self.fourier = fourier
        self.double_phase = False
        self.device = device

        self.layers = [LayerSteerablePyramid(n_i, n_j, 0, n_ori, True, 
                                             real, up_sampled, fourier,
                                             double_phase, device)]
        self.layers += [LayerSteerablePyramid(n_i, n_j, 1+k, n_ori, False,
                                              real, up_sampled, fourier, 
                                              double_phase, device)
                        for k in range(n_scale-1)]
        self.output = []
        self.low = []


    def forward(self, input):
        n_b, n_c = input.shape[:2]
        fourier_input = fftshift(fft2(input), dim=(-2,-1))

        self.output = []
        for k in range(self.n_scale):
            if k == 0:
                layer_1 = self.layers[k](fourier_input)
                self.output += [layer_1[:,:,:1],]
                self.output += [layer_1[:,:,1:-1],]
                if self.n_scale == 1:
                    self.output += [layer_1[:,:,-1:],]
            elif k == self.n_scale-1:
                layer_last = self.layers[k](fourier_input)
                self.output += [layer_last[:,:,:-1],]
                self.output += [layer_last[:,:,-1:],]
            else:
                self.output += [self.layers[k](fourier_input)[:,:,:-1],]
                
            if k < self.n_scale-1:
                fourier_input = self.layers[k].low[:,:,
                        self.layers[k+1].y.type(tch.long)+self.n_i//2**(k+1),
                        self.layers[k+1].x.type(tch.long)+self.n_j//2**(k+1)]

        self.low = [self.layers[k].output[:,:,-1:] for k in range(self.n_scale)]
        if self.up_sampled == True:
            self.output = tch.cat(self.output, dim=2)
            self.low = tch.cat(self.low, dim=2)

        return self.output

    def recompose(self):

        # update each layer with wavelet coefficients
        if self.up_sampled == True:
            out = tch.flip(self.output,(2,))
            for k in range(self.n_scale):
                if k == 0:
                    self.layers[self.n_scale-1-k].output[:,:,:-1] = tch.flip(
                        out[:,:,1+self.n_ori*k:1+self.n_ori*(k+1)],(2,))
                    self.layers[self.n_scale-1-k].output[:,:,-1:] = out[:,:,:1]         
                elif k == self.n_scale-1:
                    self.layers[self.n_scale-1-k].output[:,:,1:-1] = tch.flip(
                        out[:,:,1+self.n_ori*k:-1],(2,))
                    self.layers[self.n_scale-1-k].output[:,:,:1] = out[:,:,:1]
                else:
                    self.layers[self.n_scale-1-k].output[:,:,:-1] = tch.flip(
                        out[:,:,1+self.n_ori*k:1+self.n_ori*(k+1)],(2,))
             
        else:
            for k in range(self.n_scale):
                if k == 0:
                    self.layers[self.n_scale-1-k].output[:,:,:-1] = \
                                self.output[self.n_scale-k]
                    self.layers[self.n_scale-1-k].output[:,:,-1:] = \
                                self.output[self.n_scale+1-k]
                elif k == self.n_scale-1:
                    self.layers[self.n_scale-1-k].output[:,:,1:-1] = \
                                self.output[self.n_scale-k]
                    self.layers[self.n_scale-1-k].output[:,:,:1] = \
                                self.output[self.n_scale-1-k]
                else:
                    self.layers[self.n_scale-1-k].output[:,:,:-1] = \
                                self.output[self.n_scale-k]

        # reconstruction loop
        for k in range(self.n_scale):

            low_band = self.layers[self.n_scale-1-k].recompose()
            #disp(low_band[0,0].real)
            if k < self.n_scale-1 and self.up_sampled == False:
                pad_i = ((self.n_i//2**(self.n_scale-2-k)\
                            -self.n_i//2**(self.n_scale-1-k))//2)
                pad_j = ((self.n_j//2**(self.n_scale-2-k)\
                            -self.n_j//2**(self.n_scale-1-k))//2)
                zero_pad = tch.nn.ZeroPad2d((pad_j,pad_j,pad_i,pad_i))
                self.layers[self.n_scale-2-k].output[:,:,-1] =\
                        ifft2(ifftshift(zero_pad(low_band), dim=(-2,-1))).real
            elif k < self.n_scale-1 and self.up_sampled == True:
                self.layers[self.n_scale-2-k].output[:,:,-1] =\
                        ifft2(ifftshift(low_band, dim=(-2,-1))).real

        reconstructed_image = ifft2(ifftshift(low_band, dim=(-2,-1)))
        
        return reconstructed_image