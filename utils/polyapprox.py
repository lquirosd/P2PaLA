from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

import numpy as np

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def points_to_str(cnt):
    """
    Transform array of vertices to an string.
    In-format:      [[x1,y1],[x2,y2], ... ,[xn,yn]]
    Out-format:     x1,y1 x2,y2 ... xn,yn
    """
    return " ".join(",".join("%d" % x for x in y) for y in cnt)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def one_axis_delta(secPoints, i, j, xK, yK, xxK, yyK, xyK):
    """
    One dependent axis error measure (L2).
    Inputs:
        secPoints:  [2D-array, Nx2]     Finite (N) non-empty ardered set of coordinate pairs.
        i,j:        [int,int]           index of first and last points of the line to measure error.
    """
    # --- epsilon is added for numerical stability
    b = (secPoints[j, 1] - secPoints[i, 1]) / (
        secPoints[j, 0] - secPoints[i, 0] + np.finfo(float).eps
    )
    a = secPoints[i, 1] - (b * secPoints[i, 0])
    delta = 0
    delta = (
        (a ** 2) * (j - i - 1)
        + 2 * a * b * xK
        - 2 * a * yK
        + (b ** 2) * xxK
        + yyK
        - 2 * b * xyK
    )
    xK = xK + secPoints[i + 1, 0]
    yK = yK + secPoints[i + 1, 1]
    xxK = xxK + secPoints[i + 1, 0] ** 2
    yyK = yyK + secPoints[i + 1, 1] ** 2
    xyK = xyK + secPoints[i + 1, 0] * secPoints[i + 1, 1]
    return (xK, yK, xxK, yyK, xyK, delta)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def poly_approx(secPoints, vertM, delta):
    """
    Dynamic Programming implementation of recurence
    defined by [Perez & Vidal, 1994] on theorem 1
    'Optimum polygonal aprroximation of digitalized curves', Pattern Recognition Letters, 1994
    Inputs:
        secPoints:  [2D-array, Nx2]     finite (N) non-empty ardered set of coordinate pairs.
        vertM:      [int, 2<vertM<N]    number of vertices among "secPoints" to define the output polygon.
        delta:      [function]          a function to meassure the error (distance).
    Outputs:
        (e,v):      [float,2D-array]    tuple, e is the  measured error and v = the vertices of the polygon.
    Author:
        lquirosd, 2017
    """
    # --- if number of input point is less than M, return input
    if secPoints.shape[0] <= vertM:
        return (0.0, secPoints)

    secSize = secPoints.shape[0]
    # --- Define internal Variables
    # --- Dynamic programming matrix. The rows represent the input points and the colums the vertices.
    matD = np.zeros([secSize, vertM], dtype=np.float)
    matD.fill(np.inf)
    # --- Sequence of Vertices. Sorted in the order of secPoints
    secVec = np.zeros((vertM, 2), dtype=np.int)
    # --- Matrix to store the sequence of decisions taken
    pathMatrix = np.zeros([secSize, vertM], dtype=np.int)
    # --- Initialization of the first cell of D, because this is and endpoint bounded polygon
    matD[0, 0] = 0
    # matD[1:,0] = np.inf
    minArray = np.zeros((secSize), dtype=np.float)
    for m in range(1, vertM):
        for n in range(m, secSize):
            minArray.fill(np.inf)
            # --- Compute the cost of all nodes on current hypothesis.
            xK = yK = xxK = yyK = xyK = 0
            # for i in range(m-1,n):
            for i in range(n - 1, m - 2, -1):
                (xK, yK, xxK, yyK, xyK, minArray[i]) = delta(
                    secPoints, i, n, xK, yK, xxK, yyK, xyK
                )
                minArray[i] = minArray[i] + matD[i, m - 1]
            # --- Select only the best option
            pathMatrix[n, m] = np.argmin(minArray)
            matD[n, m] = minArray[pathMatrix[n, m]]
    # --- Retrive vertices of the polygon from the path matrix
    secVec[-1, :] = secPoints[-1]
    prev = secSize - 1
    for p in range(vertM - 1, 0, -1):
        prev = pathMatrix[prev, p]
        secVec[p - 1, :] = secPoints[prev]
    return (matD[secSize - 1, vertM - 1], secVec)


# ------------------------------------------------------------------------------


def norm_trace(sec_points, vert_m):
    """
    trace normalization algorithm
    """
    trace_long = np.zeros(sec_points.shape[0], dtype=np.float)
    output = np.zeros((vert_m, 2), dtype=int)
    p_iter = enumerate(sec_points)
    next(p_iter)
    for i, p in p_iter:
        trace_long[i] = np.sqrt(np.sum((p - sec_points[i - 1]) ** 2))
    trace_long = np.cumsum(trace_long)
    seg_long = trace_long[-1] / (vert_m - 1)
    output[0] = sec_points[0]
    n = 1
    for m in range(1, vert_m - 1):
        while not (
            (trace_long[n - 1] <= m * seg_long) and (m * seg_long <= trace_long[n])
        ):
            n += 1
        if trace_long[n - 1] == trace_long[n]:
            alpha = 1
        else:
            alpha = ((m * seg_long) - trace_long[n - 1]) / (
                trace_long[n] - trace_long[n - 1]
            )

        output[m] = sec_points[n - 1] + ((sec_points[n] - sec_points[n - 1]) * alpha)
    output[-1] = sec_points[-1]
    return output
