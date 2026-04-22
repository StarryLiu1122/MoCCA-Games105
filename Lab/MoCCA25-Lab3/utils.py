import numpy as np
import pyvista as pv

def cross_mat(v:np.ndarray) -> np.ndarray:
    """create cross-product matrix for v

    Args:
        v (np.ndarray): a vector with shape (..., 3, 1)
    """
    mat = np.stack((
        np.zeros_like(v[...,0,:]),                  -v[...,2,:],                  v[...,1,:],
                          v[...,2,:], np.zeros_like(v[...,1,:]),                 -v[...,0,:],
                         -v[...,1,:],                   v[...,0,:], np.zeros_like(v[...,2,:])
    ), axis=-1).reshape(*v.shape[:-2], 3, 3)

    return mat

def _update_boxes(x:np.ndarray, R:np.ndarray, box_meshes, box_vertices, plotter:pv.Plotter):
    num_boxes = len(box_meshes)
    pp = x.reshape(num_boxes, 1, 3, 1)
    RR = R.reshape(num_boxes, 1, 3, 3)

    box_pts = RR @ box_vertices + pp

    for box, pts in zip(box_meshes, box_pts):
        box.points = pts.reshape(-1,3)
    plotter.update(force_redraw=True)
    
    
def _plane_space(n:np.ndarray) -> np.ndarray:
    # n: (3) or (..., 3)
    # return (2, 3)    
    n = n/np.linalg.norm(n, axis=-1, keepdims=True)

    p:np.ndarray = np.zeros((*n.shape[:-1], 2, 3), dtype=n.dtype)
    
    # the first vector is determined by cross produce of n and either x-axis or z-axis
    mask = abs(n[...,2]) > 0.7 
    umask = np.logical_not(mask)
    # x_axis [1,0,0] x n
    p[mask, 0, 1] = -n[mask, 2]
    p[mask, 0, 2] = n[mask, 1]    
    # z_axis [0,0,1] x n
    p[umask, 0, 0] = -n[umask, 1]
    p[umask, 0, 1] = n[umask, 0]
        
    p[...,1,:] = np.cross(n, p[...,0,:])
    p = p/np.linalg.norm(p, axis=-1, keepdims=True)

    return p

def _diag_embed(n:np.ndarray) -> np.ndarray:
    return np.stack([np.diag(d) for d in n.reshape(-1, n.shape[-1])], axis=0).reshape(*n.shape[:-1], n.shape[-1], n.shape[-1])
        
def linear_momentum(m:np.ndarray, p:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray):
    return (m*v).sum(axis=0)
    
def angular_momentum(m:np.ndarray, I:np.ndarray, p:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray):
    m = m.reshape(-1, 1)
    Rt = R.swapaxes(-1,-2)
    I = R @ I @ Rt
    
    Iw = I @ w.reshape(-1, 3, 1)
    L = Iw.sum(axis=0).reshape(-1)
    com = ((m*p).sum(axis=0)/m.sum()).reshape(-1,3)
    L += np.cross((p-com).reshape(-1,3), (m*v).reshape(-1,3)).reshape(p.shape).sum(axis=0)
    
    return L