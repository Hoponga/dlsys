import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img: 
            return np.fliplr(img)
        return img 
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        #padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        if abs(shift_x) > img.shape[0] or abs(shift_y) > img.shape[1]:
            return np.zeros_like(img)
        
        # both shifts areguaranteed to be less than the normal dim 
        out_img = np.zeros_like(img)
        
        # Negative shift means that out img should have {0, img[0:x_max - shift_x]}, with the number of zeros as (self.padding - shift_x)
        # Positive means that out img should have {img[shift_x:x_max], 0}, with the number of zeros as (self.padding - shift_x)

        out_x_min, out_x_max = max(0, -shift_x), min(img.shape[0], -shift_x + img.shape[0])
        out_y_min, out_y_max = max(0, -shift_y), min(img.shape[1], -shift_y + img.shape[1])

        

        # If left shift, 
        img_x_min = max(0, shift_x)
        img_x_max = min(img.shape[0], shift_x + img.shape[0])
        img_y_min = max(0, shift_y)
        img_y_max = min(img.shape[1], shift_y + img.shape[1])


        out_img[out_x_min:out_x_max, out_y_min:out_y_max, :] = img[img_x_min:img_x_max, img_y_min:img_y_max, :]
        ### BEGIN YOUR SOLUTION
        return out_img
        ### END YOUR SOLUTION
