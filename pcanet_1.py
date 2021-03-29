
from chainer.cuda import to_gpu, to_cpu
from chainer.functions import convolution_2d

import numpy as np
from sklearn.decomposition import IncrementalPCA
from pcanet_func.PCANet_Func import *
from pcanet_func.utils import gpu_enabled




class PCANet1():
    def __init__(self, image_shape,
                 filter_shape_l1, step_shape_l1, n_l1_output,
                 filter_shape_pooling, step_shape_pooling):

        self.image_shape = to_tuple_if_int(image_shape)

        self.filter_shape_l1 = to_tuple_if_int(filter_shape_l1)
        self.step_shape_l1 = to_tuple_if_int(step_shape_l1)
        self.n_l1_output = n_l1_output


        self.filter_shape_pooling = to_tuple_if_int(filter_shape_pooling)
        self.step_shape_pooling = to_tuple_if_int(step_shape_pooling)
        self.n_bins = None

        self.pca_l1 = IncrementalPCA(n_l1_output)

    def histogram(self, binary_images):

        k = pow(2, self.n_l1_output)
        if self.n_bins is None:
            self.n_bins = k + 1
        bins = xp.linspace(-0.5, k - 0.5, self.n_bins)

        def bhist(image):
            # calculate Bhist(T) in the original paper
            ps = Patches(
                image,
                self.filter_shape_pooling,
                self.step_shape_pooling).patches

            H = [xp.histogram(p.flatten(), bins)[0] for p in ps]
            return xp.concatenate(H)
        return xp.vstack([bhist(image) for image in binary_images])

    def process_input(self, images):
        assert(np.ndim(images) >= 3)
        assert(images.shape[1:3] == self.image_shape)
        if np.ndim(images) == 3:
            # forcibly convert to multi-channel images
            images = atleast_4d(images)
        images = to_channels_first(images)
        return images

    def fit(self, images):
        """
        Train PCANet

        Parameters
        ----------
        images: np.ndarray
            | Color / grayscale images of shape
            | (n_images, height, width, n_channels) or
            | (n_images, height, width)
        """
        images = self.process_input(images)
        # images.shape == (n_images, n_channels, y, x)

        for image in images:
            X = []
            for channel in image:
                patches = image_to_patch_vectors(
                    channel,
                    self.filter_shape_l1,
                    self.step_shape_l1
                )
                X.append(patches)
            patches = np.hstack(X)
            # print(patches.shape)
            # patches.shape = (n_patches, n_patches * vector length)
            self.pca_l1.partial_fit(patches)
        return self

    def transform(self, images):

        images = self.process_input(images)
        # images.shape == (n_images, n_channels, y, x)

        filters_l1 = components_to_filters(
            self.pca_l1.components_,
            n_channels=images.shape[1],
            filter_shape=self.filter_shape_l1,
        )

        #
        # if gpu_enabled():
        #     images = to_gpu(images)
        #     filters_l1 = to_gpu(filters_l1)
        #     filters_l2 = to_gpu(filters_l2)

        images = convolution_2d(
            images,
            filters_l1,
            stride=self.step_shape_l1
        ).data
        #
        # return images
        # # images.shape == (n_images, L1, y, x) right here
        images = binarize(images)
        images = binary_to_decimal(images)
            # maps.shape == (n_images, y, x)
        X = self.histogram(images)# X is a set of feature vectors.

        # if gpu_enabled():
        #     X = to_gpu(X)

        X = X.astype(np.float64)

        # The shape of X is (n_images, L1 * vector length)
        return X



    def validate_structure(self):
        """
        Check that the filter visits all pixels of input images without
        dropping any information.

        Raises
        ------
        ValueError:
            if the network structure does not satisfy the above constraint.
        """
        def is_valid_(input_shape, filter_shape, step_shape):
            ys, xs = steps(input_shape, filter_shape, step_shape)
            fh, fw = filter_shape
            h, w = input_shape
            if ys[-1]+fh != h or xs[-1]+fw != w:
                print("Invalid network structure.")
            else:
                print("Valid network structure.")
            return output_shape(ys, xs)

        print("validation 1 !")
        output_shape_l1 = is_valid_(self.image_shape,
                                    self.filter_shape_l1,
                                    self.step_shape_l1)
        print(output_shape_l1)
        print("validation 2 !")
        is_valid_(
            output_shape_l1,
            self.filter_shape_pooling,
            self.step_shape_pooling
        )

