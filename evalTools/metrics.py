from __future__ import print_function
from __future__ import division
from builtins import range

import numpy as np
from page_xml.xmlPAGE import pageData
import pyclipper

# import matplotlib.pyplot as plt
import cv2

# --- GOAL Oriented Perfomance Evaluation Methodology for page Segmentation
# --- http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7333768

# --- ICDAR2009 Page Segmentation Competition Evaluation metrics
# --- http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5277763
# --- http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4377117

# --- Pixel Base P/R ICDAR2007

# ---Shoelace formula for polygon area
def poly_area(poly):
    """
    compute polygon area based on Shoelace formula(https://en.wikipedia.org/wiki/Shoelace_formula)
    code borrowed from:  https://stackoverflow.com/a/30408825/9457448
    """
    # ---  test array:
    # poly = np.array([[0,4],[0,6],[4,6],[4,8],[5,8],[5,3],[3,3],[3,1],[2,1],[2,4]])
    # area = 17
    x = poly[:, 0]  # --- x vertices
    y = poly[:, 1]  # --- y vertices
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def poly_intersect(subj, clip):
    """
    """
    pc = pyclipper.Pyclipper()
    pc.AddPath(clip, pyclipper.PT_CLIP, True)
    pc.AddPath(subj, pyclipper.PT_SUBJECT, True)
    solution = pc.Execute(
        pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD
    )
    return np.array(solution)


def matching_structure(
    hyp_poly, target_poly, epsilon=0.1, w=[0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
):
    """
    """
    img = cv2.imread(
        "/home/lquirosd/WORK/PhD/P2PaLA/to_delete/170025120000003,0611.tif"
    )
    g_len = len(target_poly)
    d_len = len(hyp_poly)
    sigma = np.zeros((g_len, d_len))
    tao = np.zeros((g_len, d_len))
    for i, t in enumerate(target_poly):
        for j, h in enumerate(hyp_poly):
            int_poly = poly_intersect(t[0], h[0])
            try:
                s = int_poly.shape[1]
            except:
                s = 0
            if s < 3:
                int_area = 0
            else:
                int_area = poly_area(int_poly[0])
            sigma[i, j] = int_area / poly_area(t[0])
            tao[i, j] = int_area / poly_area(h[0])
    # --- check for one-to-one match sigma[i,j] ~ 1 && tao[i,j] ~ 1
    o2o = np.logical_and(
        np.isclose(sigma, 1, atol=epsilon), np.isclose(tao, 1, atol=epsilon)
    ).sum()
    # --- check one to zero match sigma[i,j] ~ 0, for all j
    o2z = np.isclose(sigma.sum(axis=1), 0, atol=epsilon).sum()
    # --- check zero-to-one match tao[i,j] ~ 0, for all i
    z2o = np.isclose(tao.sum(axis=0), 0, atol=epsilon).sum()
    # --- compute f measure
    o2m = m2o = 0.0
    m2m = g_len + d_len
    for o in range(g_len):
        img = cv2.fillConvexPoly(img, target_poly[o][0], (0, 255, 0))
        # --- o2m check
        if np.all(sigma[o] < 1) and np.isclose(sigma[o].sum(), 1, epsilon):
            o2m += 1
    for o in range(d_len):
        img = cv2.fillConvexPoly(img, hyp_poly[o][0], (0, 0, 255))
        # --- m2o check
        if np.all(tao[:, o] < 1) and np.isclose(tao[:, o].sum(), 1, epsilon):
            m2o += 1
    plt.imshow(img)
    plt.show()
    m2m = m2m - 2 * o2o - o2z - o2m - m2o
    f = (np.array(w) * np.array([2 * o2o, m2o, o2m, o2z, z2o, m2m])).sum() / (
        g_len + d_len
    )
    return f


# --- Pixel level accuraccy
def pixel_accuraccy(hyp, target):
    """
    computes pixel by pixel accuraccy: sum_i(n_ii)/sum_i(t_i)
    """
    return (target == hyp).sum() / target.size


def per_class_accuraccy(hyp, target):
    """
    computes pixel by pixel accuraccy per class in target
    sum_i(n_ii/t_i)
    """
    cl = np.unique(target)
    n_cl = cl.size
    s = np.zeros(n_cl)
    # s[0] = (target[target==hyp]==0).sum()/(target==0).sum()
    for i, c in enumerate(cl):
        s[i] = (target[target == hyp] == c).sum() / (target == c).sum()
    return (s, cl)


# -- mean accuraccy
def mean_accuraccy(hyp, target):
    """
    computes mean accuraccy: 1/n_cl * sum_i(n_ii/t_i)
    """
    s, cl = per_class_accuraccy(hyp, target)
    return np.sum(s) / cl.size


def jaccard_index(hyp, target):
    """
    computes jaccard index (I/U)
    """
    cl = np.unique(target)
    n_cl = cl.size
    j_index = np.zeros(n_cl)
    for i, c in enumerate(cl):
        I = (target[target == hyp] == c).sum()
        U = (target == c).sum() + (hyp == c).sum()
        j_index[i] = I / (U - I)
    return (j_index, cl)


def mean_IU(hyp, target):
    """
    computes mean IU as defined in https://arxiv.org/pdf/1411.4038.pdf
    """
    j_index, cl = jaccard_index(hyp, target)
    return np.sum(j_index) / cl.size


def freq_weighted_IU(hyp, target):
    """
    computes frequency weighted IU as defined in https://arxiv.org/pdf/1411.4038.pdf
    """
    j_index, cl = jaccard_index(hyp, target)
    _, n_cl = np.unique(target, return_counts=True)
    return np.sum(j_index * n_cl) / target.size


# --- Lavenshtein edit distance
def levenshtein(hyp, target):
    """
    levenshtein edit distance using
    addcost=delcost=subcost=1
    Borrowed form: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(hyp) < len(target):
        return levenshtein(target, hyp)

    # So now we have len(hyp) >= len(target).
    if len(target) == 0:
        return len(hyp)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    hyp = np.array(tuple(hyp))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in hyp:
        # Insertion (target grows longer than hyp):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and hyp items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
            current_row[1:], np.add(previous_row[:-1], target != s)
        )

        # Deletion (target grows shorter than hyp):
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]
