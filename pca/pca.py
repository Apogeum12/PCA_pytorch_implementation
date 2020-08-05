from scipy.signal import correlate2d
import torch as t
import numpy as np

class LPCA:
    """
    Parameters
    ----------
    entropy: float, default, 0.35
        Say how many date delete from source in first step
    dt: float, default, 0.05
        Distance between consecutive masks. 
    """
    def __init__(self, entropy=0.35, dt=0.05, min_tolerance = 2, gpu = False):
        self.entropy = entropy
        self.dt = dt
        self.min_tolerance = min_tolerance
        if gpu:
            self.device = ("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.device = ("cpu")
    
    def _torch_standard_scaler(self, x):
        m = x.mean(0, keepdim=True).to(self.device)
        s = x.std(0, unbiased=False, keepdim=True).to(self.device)
        x -= m
        x /= s
        #del m; s
        return x
    
    def _conv_torch(self, x_std, y=None):
        if y is not None:
            x_std = t.cat((x_std, y), dim=1)
        x_std_exp = t.mean(x_std, dim=0).to(self.device)
        x = (x_std - x_std_exp).T
        cov = x.mm(x.T) / (x.size(1) - 1)
        #del X_std_exp; X
        return cov
    
    def _sort_eig_pair(self, eig_):
        _s = t.sum(t.tensor([e_[0].numpy() for e_ in eig_]))
        dim_x = len(eig_[0][1])
        k, _idx = 0.0, 0
        e_ = [eg_[0].numpy() for eg_ in eig_]
        for i, _v in enumerate(e_):
            k += _v[0]/_s
            if k >= 0.92:
                _idx = i
                break
        
        _w  =  [_w[1].reshape(dim_x, 1) for _w in eig_[0:_idx+1]]
        matrix_w  = np.hstack(_w)
        #del _s; dim_x; k; _idx; e_; _w
        return matrix_w
    
    def _sort_eig_pair_to_delete(self, eig_):
        _s = t.sum(t.tensor([e_[0].numpy() for e_ in eig_]))
        e_ = [eg_[0].numpy() for eg_ in eig_]
        
        _min_cover, _i = 0.15, None # 15% and )-index
        for i, v in enumerate(e_):
            _min = v[0]/_s
            if _min < _min_cover:
                _min_cover = _min
                _i = i
        #del _s;  e_
        return _i
    
    def _weight_mask(self, input_X, _e):
        _wei = t.ones((input_X.size(0),input_X.size(1)))
        for _ in range(int(_wei.size(0)*_wei.size(1)*_e)+1):
            _wei[np.random.randint(0, _wei.size(0)), np.random.randint(0, _wei.size(1))] = 0
        input_X = input_X * _wei
        #del _wei
        return input_X
    
    def _lpca_reduce(self, X, entropy):
        X_std = self._torch_standard_scaler(X)
        if entropy > 0.0:
                X_std = self._weight_mask(X_std, entropy)

        cov_mat = self._conv_torch(X_std)
        eig_val, eig_vecs =  t.eig(cov_mat, eigenvectors=True)
        eig_pairs = [(t.abs(eig_val[i]), eig_vecs[:,i]) for i in range(len(eig_val))]
        _delete_idx = self._sort_eig_pair_to_delete(eig_pairs)
            
        #del X_std; cov_mat; eig_val; eig_vecs; eig_pairs
        return _delete_idx

    def _lpca(self, X, entropy):
        X_std = self._torch_standard_scaler(X)
        if entropy > 0.0:
                X_std = self._weight_mask(X_std, entropy)
        
        cov_mat = self._conv_torch(X_std)
        eig_val, eig_vecs =  t.eig(cov_mat, eigenvectors=True)
        eig_pairs = [(t.abs(eig_val[i]), eig_vecs[:,i]) for i in range(len(eig_val))]
        m_w = self._sort_eig_pair(eig_pairs)
            
        #del X_std; cov_mat; eig_val; eig_vecs; eig_pairs
        return m_w
    
    def l_pca(self, X, C=False):
        entropy = self.entropy
        min_tolerance = self.min_tolerance

        if type(X) is np.ndarray:    
            X = t.from_numpy(X).to(self.device)
        list_mw = []
        
        ## Reduction Dimension with entropy ?? ->>
        size_reduce = int(X.shape[1]*0.5)
        while True:
            _delete_idx = self._lpca_reduce(X, entropy)
            if _delete_idx is None or X.shape[1] < size_reduce:
                break
            else:
                X = X.numpy()
                X_p = np.delete(X.T, _delete_idx, 0)
                X = t.from_numpy(X_p.T)
            
        # Next compute m_w for anny entropy
        while entropy >= 0:
            m_w = self._lpca(X, entropy) # Return idx to remove
            if m_w.shape[1] < min_tolerance:
                break
            list_mw.append(m_w)
            entropy -= self.dt
        
        
        if C:
            C = correlate2d(m_w[-1], m_w[-2], mode='same')
            for i in range(len(m_w)-1, 0, 2):
                if i-1 >= 0:
                    C = correlate2d(C, m_w[i], mode='same')
            return X, t.tensor(list_mw[-1]), t.tensor(C)
        else:
            return X, t.tensor(list_mw[-1])
