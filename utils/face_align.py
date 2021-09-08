# -*- coding: UTF-8 -*-
# ----------------------------------------------------------
# @Author   : Etpoem
# @Time     : 2020/2/14 14:03
# @Desc     : 
# ----------------------------------------------------------
import torch
import kornia


class AlignFace(object):
    def __init__(self):
        dst = [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2014]
        ]
        self.dst = torch.cuda.FloatTensor(dst)
        del dst

    def get(self, image, landmark):
        """
        根据人脸关键点利用仿射变换获取校正后的人脸
        :param image:          tensor -- [1, 3, h, w]
        :param landmark:       tensor -- shape [5, 2]
        :return:
                tensor -- aligned face [1, 3, 112, 112]
        """
        M = estimate_transforms(landmark, self.dst)[0:2, :]
        M = M.unsqueeze(0)
        aligned_face = kornia.warp_affine(image, M, dsize=(112, 112))
        return aligned_face


def get_align_face(image, landmark):
    """
    根据人脸关键点利用仿射变换获取校正后的人脸
    :param image:       tensor -- [b, c, h, w]
    :param landmark:    tensor -- shape [5, 2]
    :return:
            tensor -- aligned face [b, c, h, w]
    """
    dst = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2014]
    ]
    dst = torch.cuda.FloatTensor(dst)
    src = landmark
    M = estimate_transforms(src, dst)[0:2, :]
    M = M.unsqueeze(0)
    aligned_face = kornia.warp_affine(image, M, dsize=(112, 112))
    return aligned_face


def estimate_transforms(src, dst, estimate_scale=True):
    """Estimate N-D similarity transformation with or without scaling.
            Parameters
            ----------
            src : (M, N) tensor
                Source coordinates.
            dst : (M, N) tensor
                Destination coordinates.
            estimate_scale : bool
                Whether to estimate scaling factor.
            Returns
            -------
            T : (N + 1, N + 1)
                The homogeneous similarity transformation matrix. The matrix contains
                NaN values only if the problem is not well-conditioned.
            References
            ----------
            .. [1] "Least-squares estimation of transformation parameters between two
                    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
            """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)

    # Subtract mean form src and dst
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38)
    A = torch.matmul(dst_demean.T, src_demean) / num

    # Eq. (39)
    d = torch.ones(dim).cuda()
    if torch.det(A) < 0:
        d[dim - 1] = -1

    T = torch.eye(dim + 1).cuda()

    U, S, V = torch.svd(A)

    # Eq. (40) and (43)
    rank = torch.matrix_rank(A)
    if rank == 0:
        return torch.tensor(float('nan')) * T
    elif rank == dim - 1:
        if torch.det(U) * torch.det(V) > 0:
            T[:dim, :dim] = torch.matmul(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = torch.matmul(U, torch.matmul(torch.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = torch.matmul(U, torch.matmul(torch.diag(d), V.T))

    if estimate_scale:
        #  Eq. (41) and (42)
        scale = 1.0 / src_demean.var(dim=0, unbiased=False).sum() * torch.matmul(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * torch.matmul(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T
