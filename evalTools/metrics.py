from __future__ import print_function
from __future__ import division
from builtins import range

import os
import sys

import numpy as np
from page_xml.xmlPAGE import pageData
import pyclipper

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from page_xml.xmlPAGE import pageData

import matplotlib.pyplot as plt
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

def area_bin(poly, img):
    """
    returns the number of black pixels insude a polygon
    """
    


def zone_map(hyp, target, img, alpha=0, dist=lambda r,h:int(r==h)):
    """
    Computes ZoneMap metric from:
    O. Galibert, J. Kahn and I. Oparin, The zonemap metric for page 
    segmentation and area classification in scanned documents, 2014 
    IEEE International Conference on Image Processing (ICIP), Paris, 
    2014, pp. 2594-2598.
    Inputs: 
        hyp: hypoteses data (PAGE-XML object)
        target: reference data (PAGE-XML object)
        img: pointer to image file. If img is not binary img=Otsu(img) 
        alpha: classification error weigth [0,1]
        dist: difference function between zones [0,1], default:
            dist(h,r)=lambda(0 if h==r; 1 else)
    """
    if os.path.isfile(img):
        img = cv2.imread(img,0)
    if np.max(img) > 1:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = np.abs(1-img).astype(np.uint8)

    hyp_data = pageData(hyp)
    hyp_data.parse()
    tar_data = pageData(target)
    tar_data.parse()
    Hzones = hyp_data.get_zones(["TextRegion"])
    Rzones = tar_data.get_zones(["TextRegion"])
    #--- Mapping
    tmp_H = np.zeros(img.shape,dtype=np.uint8)
    tmp_R = np.zeros(img.shape,dtype=np.uint8)
    force = np.zeros((len(Rzones),len(Hzones)))
    area_R = 0
    for i,R in enumerate(sorted(Rzones.keys())):
        tmp_R.fill(0)
        cv2.fillConvexPoly(tmp_R, Rzones[R]['coords'].astype(np.int), 1)
        Rzones[R]['area'] = np.logical_and(img==1, img==tmp_R).sum()
        area_R += Rzones[R]['area']
        for j,H in enumerate(sorted(Hzones.keys())):
            tmp_H.fill(0)
            cv2.fillConvexPoly(tmp_H, Hzones[H]['coords'].astype(np.int),1)
            #--- Intersection 
            I = img[np.where(np.logical_and(tmp_R == tmp_H,tmp_R ==1))].sum()
            #I = img[np.where(tmp_R == tmp_H)].sum()
            Hzones[H]['area'] = np.logical_and(tmp_H==1, img==1).sum()
            force[i][j] = I*((1/Rzones[R]['area'])+(1/Hzones[H]['area']))

    print(force)
    #--- get and sort non-zero links
    nz = force != 0
    s_index = np.unravel_index(np.where(nz, force, np.nan).argsort(axis=None)[:nz.sum()],force.shape)
    s_index= (s_index[0][::-1], s_index[1][::-1])
    #--- make groups
    #--- Manage False Alarm and Miss groups
    #---    False Alarm:
    fa = np.where(force.sum(axis=0)==0)
    g_id = 1
    G = {}
    for f in fa[0]:
        G[g_id] = ([],[f])
        #G[g_id] = ([],[Hzones[f]['id']])
        g_id += 1
    #---    Miss
    mi = np.where(force.sum(axis=1)==0)
    L = (np.zeros(len(Rzones), dtype=np.uint8),np.zeros(len(Hzones),dtype=np.uint8))
    for m in mi[0]:
        G[g_id] = ([m],[])
        #G[g_id] = ([Rzones[m]['id']],[])
        g_id += 1
    for r_ix,h_ix in zip(s_index[0],s_index[1]):
        print("H:{} R:{} L:{}".format(Hzones[h_ix]['id'],Rzones[r_ix]['id'],force[r_ix,h_ix]))
        if L[0][r_ix] > 0 and L[1][h_ix] > 0:
            #--- do not add to any group
            pass
        elif L[0][r_ix] == 0 and L[1][h_ix] ==0:
            #--- create new group
            G[g_id] = ([r_ix],([h_ix]))
            #G[g_id] = ([Rzones[r_ix]['id']],[Hzones[h_ix]['id']])
            L[0][r_ix] = g_id
            L[1][h_ix] = g_id
            g_id += 1
        elif L[0][r_ix] == 0:
            #--- only ref is not assigned
            p_id = L[1][h_ix]
            G[p_id][0].append(r_ix)
            #G[p_id][0].append(Rzones[r_ix]['id'])
            L[0][r_ix] = p_id
            
        else:
            #--- only hyp is not assigned
            p_id = L[0][r_ix]
            G[p_id][1].append(h_ix)
            #G[p_id][1].append(Hzones[h_ix]['id'])
            L[1][h_ix] = p_id
    print(G)
    #--- Second stage, error calcualtion 
    Ezm = 0
    for gr_key in G.keys():
        #--- handle false alarm:
        if len(G[gr_key][0]) == 0:
            Es = Hzones[G[gr_key][1][0]]['area']
            #--- Ec = Es, E=(1-a)Es + aEc == Ec 
            Ezm += Es
        #--- handle miss:
        elif len(G[gr_key][1]) == 0:
            Es = Rzones[G[gr_key][0][0]]['area']
            Ezm += Es
        #--- handle match
        elif len(G[gr_key][0]) == 1 and len(G[gr_key][1]) == 1:
            tmp_R.fill(0)
            cv2.fillConvexPoly(tmp_R, Rzones[G[gr_key][0][0]]['coords'].astype(np.int), 1)
            tmp_H.fill(0)
            cv2.fillConvexPoly(tmp_H, Hzones[G[gr_key][1][0]]['coords'].astype(np.int), 1)
            reg_intersection = (img*tmp_R)[np.where(tmp_R == tmp_H)].sum()
            Es = Rzones[G[gr_key][0][0]]['area'] + Hzones[G[gr_key][1][0]]['area'] - \
                    (2* reg_intersection)
            Ec = (dist(Rzones[G[gr_key][0][0]]['type'],Hzones[G[gr_key][1][0]]['type']) * reg_intersection) + Es
            Ezm += ((1-alpha)*Es)+(alpha*Ec)
        #--- handle split
        elif len(G[gr_key][0]) == 1 and len(G[gr_key][1]) > 1:
            tmp_R.fill(0)
            cv2.fillConvexPoly(tmp_R, Rzones[G[gr_key][0][0]]['coords'].astype(np.int), 1)
            tmp_H.fill(0)
            tmp_H_o = tmp_H.copy()
            for i,h in enumerate(G[gr_key][1]):
                print(h)
                cv2.fillConvexPoly(tmp_H, Hzones[h]['coords'].astype(np.int), 1 + i)
                cv2.fillConvexPoly(tmp_H_o, Hzones[h]['coords'].astype(np.int), 1)
            #--- handle sub-zones by configuration
            #--- Miss Error i.e. in R but not in any H
            #--- E = Area(z)
            Er = (img*tmp_R)[np.where(tmp_R != tmp_H_o)].sum()
            #--- False detection error i.e. H but not in R
            Er += (img*tmp_H_o)[np.where(tmp_R != tmp_H_o)].sum()
            #--- Segmentation error and Correct parts
            Ec = 0 



        #--- handle merge
        elif len(G[gr_key][0]) > 1 and len(G[gr_key][1]) == 1:
            pass


    print(Ezm/area_R)

        

    


    


    

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
    smooth = np.finfo(np.float).eps
    cl = np.unique(target)
    n_cl = cl.size
    j_index = np.zeros(n_cl)
    for i, c in enumerate(cl):
        I = (target[target == hyp] == c).sum()
        U = (target == c).sum() + (hyp == c).sum()
        j_index[i] = (I + smooth) / (U - I + smooth)
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
