import numpy as np
import cv2


class ContourCheckingFn(object):
    # Defining __call__ method
    def __call__(self, pt):
        raise NotImplementedError


class isInContourV1(ContourCheckingFn):
    def __init__(self, contour):
        self.cont = contour

    def __call__(self, pt):
        return (
            1
            if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False)
            >= 0
            else 0
        )


class isInContourV2(ContourCheckingFn):
    def __init__(self, contour, patch_size):
        self.cont = contour
        self.patch_size = patch_size

    def __call__(self, pt):
        pt = np.array(
            (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        ).astype(float)
        return (
            1
            if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False)
            >= 0
            else 0
        )


# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(ContourCheckingFn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift),
            ]
        else:
            all_points = [center]

        for points in all_points:
            if (
                cv2.pointPolygonTest(
                    self.cont, tuple(np.array(points).astype(float)), False
                )
                >= 0
            ):
                return 1
        return 0


# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class isInContourV3_Hard(ContourCheckingFn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift),
            ]
        else:
            all_points = [center]

        for points in all_points:
            if (
                cv2.pointPolygonTest(
                    self.cont, tuple(np.array(points).astype(float)), False
                )
                < 0
            ):
                return 0
        return 1
