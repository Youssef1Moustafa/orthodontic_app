import numpy as np
import cv2

class ThinPlateSpline:
    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.src_points = None
        self.dst_points = None
        self.weights = None
        
    def fit(self, src, dst):
        self.src_points = src.copy()
        self.dst_points = dst.copy()
        n = src.shape[0]
        
        K = self._rbf(src, src)
        P = np.hstack([np.ones((n, 1)), src])
        
        M = np.zeros((n + 3, n + 3))
        M[:n, :n] = K + self.alpha * np.eye(n)
        M[:n, n:] = P
        M[n:, :n] = P.T
        
        Y = np.vstack([dst, np.zeros((3, 2))])
        self.weights = np.linalg.solve(M, Y)
        
    def transform(self, points):
        n = points.shape[0]
        K = self._rbf(points, self.src_points)
        P = np.hstack([np.ones((n, 1)), points])
        return K @ self.weights[:len(self.src_points)] + P @ self.weights[len(self.src_points):]
    
    def _rbf(self, x, y):
        dist = np.sqrt(((x[:, None] - y[None, :])**2).sum(axis=2))
        return dist**2 * np.log(dist + 1e-10)

def warp_face_tps(image, src_landmarks, dst_landmarks):
    h, w = image.shape[:2]
    
    src = src_landmarks.astype(np.float32)
    dst = dst_landmarks.astype(np.float32)
    
    src = np.clip(src, [0, 0], [w-1, h-1])
    dst = np.clip(dst, [0, 0], [w-1, h-1])
    
    try:
        corners = np.array([
            [0,0], [w-1,0], [0,h-1], [w-1,h-1],
            [w//2,0], [w//2,h-1], [0,h//2], [w-1,h//2]
        ], dtype=np.float32)
        
        src_all = np.vstack([src, corners])
        dst_all = np.vstack([dst, corners])
        
        tps = ThinPlateSpline(alpha=9600.0)
        tps.fit(dst_all, src_all)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        
        warped_grid = tps.transform(grid)
        warped_grid = np.clip(warped_grid, [0, 0], [w-1, h-1])
        
        map_x = warped_grid[:,0].reshape(h,w).astype(np.float32)
        map_y = warped_grid[:,1].reshape(h,w).astype(np.float32)
        
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        return warped
        
    except Exception as e:
        print(f"TPS error: {e}")
        return image