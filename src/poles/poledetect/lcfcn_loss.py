# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Loss functions for LCFCN loss obtained from https://github.com/ElementAI/LCFCN/blob/master/lcfcn/lcfcn_loss.py.
This is a multicomponent loss including image level, point level, split level, and false positive level losses."""
# Necessary imports
import numpy as np
import skimage
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import morphology as morph
from skimage.segmentation import find_boundaries, watershed


def compute_loss(points, probs, roi_mask=None):
    """
    images: n x c x h x w
    probs: h x w (0 or 1)
    """
    points = points.squeeze()
    probs = probs.squeeze()

    assert points.max() <= 1

    tgt_list = get_tgt_list(points, probs, roi_mask=roi_mask)

    # image level
    pr_flat = probs.view(-1)

    # compute loss
    loss = 0.0
    for tgt_dict in tgt_list:
        pr_subset = pr_flat[tgt_dict["ind_list"]]
        loss += tgt_dict["scale"] * F.binary_cross_entropy(
            pr_subset,
            torch.ones(pr_subset.shape, device=pr_subset.device) * tgt_dict["label"],
            reduction="mean",
        )

    return loss


@torch.no_grad()
def get_tgt_list(points, probs, roi_mask=None):
    """_summary_

    Args:
        points (_type_): _description_
        probs (_type_): _description_
        roi_mask (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    tgt_list = []

    # image level
    pt_flat = points.view(-1)
    pr_flat = probs.view(-1)

    u_list = points.unique()
    if 0 in u_list:
        ind_bg = pr_flat.argmin()
        tgt_list += [{"scale": 1, "ind_list": [ind_bg], "label": 0}]

    if 1 in u_list:
        ind_fg = pr_flat.argmax()
        tgt_list += [{"scale": 1, "ind_list": [ind_fg], "label": 1}]

    # point level
    if 1 in u_list:
        ind_fg = torch.where(pt_flat == 1)[0]
        tgt_list += [{"scale": len(ind_fg), "ind_list": ind_fg, "label": 1}]

    # get blobs
    probs_numpy = probs.detach().cpu().numpy()
    blobs = get_blobs(probs_numpy, roi_mask=None)

    # get foreground and background blobs
    points = points.cpu().numpy()
    fg_uniques = np.unique(blobs * points)
    bg_uniques = [x for x in np.unique(blobs) if x not in fg_uniques]

    # split level
    # -----------
    n_total = points.sum()

    if n_total > 1:
        # global split
        boundaries = watersplit(probs_numpy, points)
        ind_bg = np.where(boundaries.ravel())[0]

        tgt_list += [{"scale": (n_total - 1), "ind_list": ind_bg, "label": 0}]

        # local split
        for u in fg_uniques:
            if u == 0:
                continue

            ind = blobs == u

            b_points = points * ind
            n_points = b_points.sum()

            if n_points < 2:
                continue

            # local split
            boundaries = watersplit(probs_numpy, b_points) * ind
            ind_bg = np.where(boundaries.ravel())[0]

            tgt_list += [{"scale": (n_points - 1), "ind_list": ind_bg, "label": 0}]

    # fp level
    for u in bg_uniques:
        if u == 0:
            continue

        b_mask = blobs == u
        if roi_mask is not None:
            b_mask = roi_mask * b_mask
        if b_mask.sum() == 0:
            pass
        else:
            ind_bg = np.where(b_mask.ravel())[0]
            tgt_list += [{"scale": 1, "ind_list": ind_bg, "label": 0}]

    return tgt_list


def watersplit(_probs, _points):
    """Implements watershed algorithm to split blobs

    Args:
        _probs (array): predicted blob probabilities
        _points (array): pole points

    Returns:
        watershed boundaries : find and returns boundaries of watershed
    """
    points = _points.copy()

    points[points != 0] = np.arange(1, points.sum() + 1)
    points = points.astype(float)

    probs = ndimage.black_tophat(_probs.copy(), 7)
    seg = watershed(probs, points)

    return find_boundaries(seg)


def get_blobs(probs, roi_mask=None):
    """Find blobs in a probability map

    Args:
        probs (array): Probability map
        roi_mask (array, optional): Region of interest for blob search. Defaults to None.

    Returns:
        _type_: _description_
    """
    h, w = probs.shape

    pred_mask = (probs > 0.5).astype("uint8")
    blobs = np.zeros((h, w), int)

    blobs = morph.label(pred_mask == 1)

    if roi_mask is not None:
        blobs = (blobs * roi_mask[None]).astype(int)

    return blobs


def blobs2points(blobs):
    """Convert blobs to points

    Args:
        blobs (array): Array of blobs

    Returns:
        points (array): Array with blob centroids
    """
    blobs = blobs.squeeze()
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs)

    assert points.ndim == 2

    for r in rps:
        y, x = r.centroid

        points[int(y), int(x)] = 1

    return points


def get_random_points(mask, n_points=1, seed=1):
    """Select n random points

    Args:
        mask (array): Input mask to select points from
        n_points (int, optional): Number of points to select . Defaults to 1.
        seed (int, optional):  Seed for random selection. Defaults to 1.

    Returns:
        point (array): Array with selected points
    """
    from haven import haven_utils as hu

    y_list, x_list = np.where(mask)
    points = np.zeros(mask.squeeze().shape)
    with hu.random_seed(seed):
        for i in range(n_points):
            yi = np.random.choice(y_list)
            x_tmp = x_list[y_list == yi]
            xi = np.random.choice(x_tmp)
            points[yi, xi] = 1

    return points


def get_points_from_mask(mask, bg_points=0):
    """Retrieve set of points from mask

    Args:
        mask (array): Input mask to select points from
        bg_points (int, optional): Number of background points to retrieve. Defaults to 0.

    Returns:
        points (array): Array with selected points
    """
    n_points = 0
    points = np.zeros(mask.shape)
    assert len(np.setdiff1d(np.unique(mask), [0, 1, 2])) == 0

    for c in np.unique(mask):
        if c == 0:
            continue
        blobs = morph.label((mask == c).squeeze())
        points_class = blobs2points(blobs)

        ind = points_class != 0
        n_points += int(points_class[ind].sum())
        points[ind] = c
    assert morph.label((mask).squeeze()).max() == n_points
    points[points == 0] = 255
    if bg_points == -1:
        bg_points = n_points

    if bg_points:
        from haven import haven_utils as hu

        y_list, x_list = np.where(mask == 0)
        with hu.random_seed(1):
            for i in range(bg_points):
                yi = np.random.choice(y_list)
                x_tmp = x_list[y_list == yi]
                xi = np.random.choice(x_tmp)
                points[yi, xi] = 0

    return points
