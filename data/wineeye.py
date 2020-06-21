import os.path
from dataset import Dataset

class WineEye(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(WineEye, self).__init__(root, train, transform)
        print "Loaded {} samples for dataset {},  {} classes, {} instances".format(len(self), self.name, self.num_cls, self.num_instance)

    @property
    def name(self):
        return 'wineeye_{}'.format('train' if self.train else 'test')

    @property
    def image_root_dir(self):
        return self.root

    @property
    def num_cls(self):
        return len(self.class_map)

    @property
    def num_instance(self):
        return len(self.instance_map)

    def _load(self):
        self.class_map = {}
        self.instance_map = {}

        for wine_id in os.listdir(self.root):
            if wine_id[0] != '1':
                if self.train:
                    continue
            elif not self.train:
                continue
            self.class_map[wine_id] = len(self.class_map)
            self.instance_map[wine_id] = len(self.instance_map)
    
            dir = os.path.join(self.root, wine_id)
            for img in os.listdir(dir):
                self.image_paths.append(os.path.join(dir, img))
                self.class_labels.append(self.class_map[wine_id])
                self.instance_labels.append(self.instance_map[wine_id])
    
