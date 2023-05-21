import numpy as np
from sklearn.linear_model import LinearRegression


def fit_line_ransac(points, max_iter=200, sigma=0.03):
    if len(points) < 2:
        return None, None
    best_score, best_inlier_mask = 0, []
    for _ in range(max_iter):
        p1_idx, p2_idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[p1_idx], points[p2_idx]
        # compute the distance from a point to the line by cross product
        line_dir = p1 - p2
        line_dir /= np.linalg.norm(line_dir)
        v = points - p2
        dis = np.abs(v.dot(
            np.array([[line_dir[1]], [-line_dir[0]]])
        )).squeeze()
        inlier_mask = dis <= sigma
        score = np.sum(inlier_mask)
        if best_score < score:
            best_score = score
            best_inlier_mask = inlier_mask
    inliers = points[best_inlier_mask]
    reg = LinearRegression().fit(inliers[:, :1], inliers[:, 1])
    coef = [reg.coef_, reg.intercept_]
    return coef, best_inlier_mask
