import os
import csv
import torch

class LocalShapeLoader():

    def __init__(self, csv_path):
        assert os.path.isfile(csv_path), 'Given CSV_PATH does not exist: %s!' % csv_path
        csv_path = os.path.abspath(csv_path)
        self.dir_path = os.path.dirname(csv_path)

        with open(csv_path, 'r') as f:
            # remove comments within csv
            f = filter(lambda row: row[0] != '#', f)
            rows = csv.DictReader(f, delimiter=',')
            self.rows = [row for row in rows]

        mod_names = self.get_vector_names()
        print('Found the following local shape modifications:')
        print(mod_names)

    def load_vector(self, name, pytorch=False):
        row = self._get_vector_info(name)
        ckpt_path = os.path.join(self.dir_path,
                                 row['path'])
        d_alpha = torch.load(ckpt_path).detach()
        if not pytorch:
            return d_alpha.numpy()
        else:
            return d_alpha


    def _get_vector_info(self, name):
        for row in self.rows:
            if row['name'] == name:
                return row
        else:
            raise Exception('No local shape vector found with the following name: %s' % name)


    def get_factor_range(self, name):
        row = self._get_vector_info(name)
        f_min = float(row['min_factor'])
        f_max = float(row['max_factor'])
        return (f_min, f_max)

    def get_vector_names(self):
        return [row['name'] for row in self.rows]

