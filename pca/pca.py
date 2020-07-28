from scipy.signal import correlate2d
import torch as t
import numpy as np

#TODO: pytorch Device-{cpu, cuda}

class LPCA:
    def __init__(self, entropy=0.35, dt=0.05, min_tolerance = 2):
        self.entropy = entropy
        self.dt = dt
        self.min_tolerance = min_tolerance
    
    def __torch_standard_scaler__(self, x):
        m = x.mean(0, keepdim=True)
        s = x.std(0, unbiased=False, keepdim=True)
        x -= m
        x /= s
        #del m; s
        return x
    
    def __conv_torch__(self, x_std, y=None):
        if y is not None:
            x_std = t.cat((x_std, y), dim=1)
        x_std_exp = t.mean(x_std, dim=0)
        x = (x_std - x_std_exp).T
        cov = x.mm(x.T) / (x.size(1) - 1)
        #del X_std_exp; X
        return cov
    
    def __sort_eig_pair__(self, eig_):
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
    
    def __sort_eig_pair_to_delete__(self, eig_):
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
    
    def __weight_mask__(self, input_X, _e):
        _wei = t.ones((input_X.size(0),input_X.size(1)))
        for _ in range(int(_wei.size(0)*_wei.size(1)*_e)+1):
            _wei[np.random.randint(0, _wei.size(0)), np.random.randint(0, _wei.size(1))] = 0
        input_X = input_X * _wei
        #del _wei
        return input_X
    
    def __l_pca_reduce__(self, X, entropy):
        X_std = self.__torch_standard_scaler__(X)
        if entropy > 0.0:
                X_std = self.__weight_mask__(X_std, entropy)

        cov_mat = self.__conv_torch__(X_std)
        eig_val, eig_vecs =  t.eig(cov_mat, eigenvectors=True)
        eig_pairs = [(t.abs(eig_val[i]), eig_vecs[:,i]) for i in range(len(eig_val))]
        _delete_idx = self.__sort_eig_pair_to_delete__(eig_pairs)
            
        #del X_std; cov_mat; eig_val; eig_vecs; eig_pairs
        return _delete_idx

    def __l_pca__(self, X, entropy):
        X_std = self.__torch_standard_scaler__(X)
        if entropy > 0.0:
                X_std = self.__weight_mask__(X_std, entropy)
        
        cov_mat = self.__conv_torch__(X_std)
        eig_val, eig_vecs =  t.eig(cov_mat, eigenvectors=True)
        eig_pairs = [(t.abs(eig_val[i]), eig_vecs[:,i]) for i in range(len(eig_val))]
        m_w = self.__sort_eig_pair__(eig_pairs)
            
        #del X_std; cov_mat; eig_val; eig_vecs; eig_pairs
        return m_w
    
    def l_pca(self, X, min_tolerance, C=False):
        entropy = self.entropy
        min_tolerance = self.min_tolerance

        if type(X) is np.ndarray:    
            X = t.from_numpy(X)
        list_mw = []
        
        ## Reduction Dimension with entropy ?? ->>
        size_reduce = int(X.shape[1]*0.5)
        while True:
            _delete_idx = self.__l_pca_reduce__(X, entropy)
            if _delete_idx is None or X.shape[1] < size_reduce:
                break
            else:
                X = X.numpy()
                X_p = np.delete(X.T, _delete_idx, 0)
                X = t.from_numpy(X_p.T)
            
        # Next compute m_w for anny entropy
        while entropy >= 0:
            m_w = self.__l_pca__(X, entropy) # Return idx to remove
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
