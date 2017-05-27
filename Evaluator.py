
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import numpy as np

def evaluator(batch_size, ref_image_set, output_image_set):
    psnr_val = psnr(batch_size, ref_image_set, output_image_set)
    ssim_val = ssim(batch_size, ref_image_set, output_image_set)

    return psnr_val, ssim_val

def psnr(batch_size, ref_image_set, output_image_set):
    psnr_val = 0
    ref_image_set = np.array(ref_image_set, dtype=np.uint8)
    output_image_set = np.array(output_image_set, dtype=np.uint8)

    if batch_size == 1:
        if ref_image_set.ndim > 3:
            ref_image_set = ref_image_set[0]
            output_image_set  = output_image_set[0]
        return compare_psnr(ref_image_set, output_image_set, data_range=ref_image_set.max() - ref_image_set.min())

    for i in range(batch_size):
        ref_image = ref_image_set[i]
        output_image = output_image_set[i]
        psnr_val += compare_psnr(ref_image, output_image, data_range=ref_image.max() - ref_image.min())
    if batch_size is 0:
        print("batch size is zero error!")
    else:
        psnr_val /= batch_size
    return psnr_val


def ssim(batch_size, ref_image_set, output_image_set):
    ssim_val = 0

    ref_image_set = np.array(ref_image_set, dtype=np.uint8)
    output_image_set = np.array(output_image_set, dtype=np.uint8)

    if batch_size == 1:
        if ref_image_set.ndim > 3:
            ref_image_set = ref_image_set[0]
            output_image_set = output_image_set[0]
        return compare_ssim(ref_image_set, output_image_set,
                            data_range=ref_image_set.max() - ref_image_set.min(), multichannel=True)

    for i in range(batch_size):
        ref_image = ref_image_set[i]
        output_image = output_image_set[i]
        ssim_val += compare_ssim(ref_image, output_image, data_range=ref_image.max() - ref_image.min(),
                                 multichannel=True)
    if batch_size is 0:
        print("batch size is zero error!")
    else:
        ssim_val /= batch_size

    return ssim_val



