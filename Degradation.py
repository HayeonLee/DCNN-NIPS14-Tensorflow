from skimage import exposure, filters, io, img_as_float, img_as_ubyte, transform, util
import numpy as np
import matplotlib.pyplot as plt
import warnings

def degrade(image, degrade_type):
    degrade_image = img_as_float(image, force_copy=True)

    if ("blur" in degrade_type):
        # K = fspecial('disk', 7) ; circular averaging filter
        # IBlur = imfilter(I, K, 'conv', 'replicate');
        # IBlur = uint8((IBlur)*255+0.5);
        # IBlur = imnoise(IBlur, 'gaussian', 0, 0.0005);
        degrade_image = filters.gaussian(degrade_image, sigma=7, multichannel=True)


    if ("noise" in degrade_type):
        degrade_image = util.random_noise(degrade_image, mode='gaussian', clip=True, mean=0, var=0.0000005)


    if ("saturate" in degrade_type):
        # I = im2double(I) * 1.3;
        degrade_image = degrade_image * 1.1
        degrade_image = exposure.rescale_intensity(degrade_image, in_range=(0, 1))


    if("downscale" in degrade_type):
        # downscale by 4
        scale = 1
        '''
        H, W, C = degrade_image.shape
        crop_H = H//scale
        crop_W = W//scale
        left = crop_W
        right = W - crop_W
        top = crop_H
        bottom = H - crop_H
        print(W, H, left, right, top, bottom)
        degrade_image = degrade_image[top:bottom, left:right, :]
        '''
        # order = 3 -> bi-cubic
        degrade_image = transform.rescale(degrade_image, (1./scale), order=3, mode='constant')
        degrade_image = transform.rescale(degrade_image, (scale), order=3, mode='constant')
        
        

   
    if ("compress" in degrade_type):
        # % jpeg compression
        # imwrite(IBlur, 'blur.jpg', 'Quality', 70);
    	
    	with warnings.catch_warnings():
	     	warnings.simplefilter("ignore")
	     	io.imsave("images/compress.jpeg", degrade_image, plugin='pil', quality=95)
	     	degrade_image = io.imread("images/compress.jpeg")
	     	
	
    with warnings.catch_warnings():
    	warnings.simplefilter("ignore")	     	
    	degrade_image = img_as_ubyte(degrade_image)
   	
    return degrade_image


'''
def main():
    fig, axes = plt.subplots(nrows=1, ncols=8,
                             sharex=True, sharey=True)
    ax = axes.ravel()

    image = io.imread("images/train/Mars.jpeg")
    ax[0].imshow(img_as_float(image))
    blur_image = degrade(image, ["blur"])
    ax[1].imshow(blur_image)
    saturate_image = degrade(image, ["saturate"])
    ax[2].imshow(saturate_image)
    downscale_image = degrade(image, ["downscale"])
    ax[3].imshow(downscale_image)
    compress_image = degrade(image, ["compress"])
    ax[4].imshow(compress_image)
    noise_image1 = degrade(image, ['blur','noise','saturate','compress'])
    ax[5].imshow(noise_image1)
    noise_image2 = degrade(image, ['noise','saturate','compress'])
    ax[6].imshow(noise_image2)
    noise_image3 = degrade(image, ['noise', 'downscale', 'compress'])
    ax[7].imshow(noise_image3)
    plt.tight_layout()
    plt.show()


main()
'''
