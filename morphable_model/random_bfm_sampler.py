import os
import numpy as np

stat_list = ['alpha_exp.npy',
             'alpha.npy',
             'angles.npy',
             'f.npy',
             't3d.npy']
STATS_PATH = 'data/configs/300W_LP_stats'

class BFMSampler():
    def __init__(self):

        self.thesh = 0.05

        self.alpha_thresh = list()
        self.alpha_exp_thresh = list()
        self.transformation_thresh = dict()

        for f in stat_list:
            path = os.path.join(STATS_PATH, f)
            stats = np.load(path)

            if 'alpha.npy' == f:
                stats = stats[:, :, 0].swapaxes(0, 1)
                stats.sort(axis=1)
                len_pc = stats.shape[1]
                min_stats = stats[:, int(self.thesh*len_pc)]
                max_stats = stats[:, int((1-self.thesh)*len_pc)]
                self.alpha_thresh = [min_stats, max_stats]
            elif 'alpha_exp.npy' == f:
                self.alpha_exp = stats[:, :, 0].swapaxes(0, 1)

            elif f == 'f.npy':
                stats.sort()
                len_pc = stats.shape[0]
                min_stats = stats[int(self.thesh*len_pc)]
                max_stats = stats[int((1-self.thesh)*len_pc)]
                self.transformation_thresh['f'] = [min_stats, max_stats]
            elif f == 't3d.npy':
                stats = stats[:, :, 0].swapaxes(0, 1)
                stats.sort(axis=1)
                len_pc = stats.shape[1]
                min_stats = stats[:, int(self.thesh*4*len_pc)]
                max_stats = stats[:, int((1-self.thesh*4)*len_pc)]
                for l, label in enumerate(['x', 'y', 'z']):
                    self.transformation_thresh[f't3d_{label}'] = [min_stats[l],
                                                                  max_stats[l]]
            elif f == 'angles.npy':
                stats = stats.swapaxes(0, 1)
                stats.sort(axis=1)
                len_pc = stats.shape[1]
                min_stats = stats[:, int(self.thesh*len_pc)]
                max_stats = stats[:, int((1-self.thesh)*len_pc)]
                for l, label in enumerate(['x', 'y', 'z']):
                    self.transformation_thresh[f'angles_{label}'] = [min_stats[l],
                                                                     max_stats[l]]

    def get_random_face(self):
        a_low, a_high = self.alpha_thresh
        alpha = np.random.uniform(a_low, a_high, size=(199))

        # Apply similar distribution to tex to account for missing ev values
        tex = np.random.uniform(a_low, a_high, size=(199))
        tex = tex / np.max(tex)

        return alpha, tex

    def get_random_expression(self):
        num_expressions = self.alpha_exp.shape[1]
        i_exp = np.random.randint(num_expressions)

        return(self.alpha_exp[:, i_exp])

    def get_random_transform(self):
        low, high = self.transformation_thresh['f']
        f = np.random.uniform(low, high)

        low, high = self.transformation_thresh['angles_x']
        ang_x = np.random.uniform(low, high)
        low, high = self.transformation_thresh['angles_y']
        ang_y = np.random.uniform(low, high)
        low, high = self.transformation_thresh['angles_z']
        ang_z = np.random.uniform(low, high)

        low, high = self.transformation_thresh['t3d_x']
        mean_x = (low+high)/2
        if ang_y > 0:
            low = mean_x
            high = mean_x + 25
        else:
            high = mean_x
            low = mean_x - 25
        x = np.random.uniform(low, high)
        low, high = self.transformation_thresh['t3d_y']
        y = np.random.uniform(low, high)
        low, high = self.transformation_thresh['t3d_z']
        z = np.random.uniform(low, high)


        return f, np.asarray([ang_x, ang_y, ang_z]), np.asarray([x, y, z])
