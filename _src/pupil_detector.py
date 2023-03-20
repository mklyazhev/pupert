import os
import multiprocessing
import itertools
import dlib
from eye_detector import EyesDetector
import utils


class PupilsDetector:
    def __init__(self, model_save_path=None, tree_depth=4, nu=0.1, cascade_depth=15, feature_pool_size=400,
                 num_test_splits=50, oversampling_amount=5, oversampling_translation_jitter=0.1, verbose=False):
        self.model = None
        self.model_path = model_save_path if model_save_path else './pupils_detector.dat'
        self.cpu_count = multiprocessing.cpu_count()
        self.tree_depth = tree_depth
        self.nu = nu
        self.cascade_depth = cascade_depth
        self.feature_pool_size = feature_pool_size
        self.num_test_splits = num_test_splits
        self.oversampling_amount = oversampling_amount
        self.oversampling_translation_jitter = oversampling_translation_jitter
        self.eye_detector = EyesDetector()
        self.verbose = verbose

    def fit(self, train_xml_path, cpu_usage=None):
        if self.verbose:
            print('[INFO] Setting shape predictor options...')
        options = dlib.shape_predictor_training_options()
        options.tree_depth = self.tree_depth
        options.nu = self.nu
        options.cascade_depth = self.cascade_depth
        options.feature_pool_size = self.feature_pool_size
        options.num_test_splits = self.num_test_splits
        options.oversampling_amount = self.oversampling_amount
        options.oversampling_translation_jitter = self.oversampling_translation_jitter
        options.be_verbose = self.verbose
        options.num_threads = cpu_usage if cpu_usage else self.cpu_count
        if self.verbose:
            print('[INFO] Pupils detector options:')
            print(options)
            print('[INFO] Training shape predictor...')
        self.__del_exist_model(self.model_path)
        dlib.train_shape_predictor(train_xml_path, self.model_path, options)
        self.model = dlib.shape_predictor(self.model_path)

    def predict(self, gray_img, rect):
        shape = self.model(gray_img, rect)
        shape = utils.shape_to_np(shape)
        px, py = shape[0][0], shape[0][1]
        return (px, py)

    def eval(self, test_xml_path):
        if self.verbose:
            print('[INFO] Evaluating shape predictor...')
        error = dlib.test_shape_predictor(test_xml_path, self.model_path)
        if self.verbose:
            print(f'[INFO] Error: {error}')
        return error

    def grid_search(self, train_xml_path, test_xml_path, params_dict, get_best=False):
        if self.verbose:
            n_trains = len([t for t in itertools.product(*params_dict.values())])
            n = 0
        res = {}
        for tree_depth in params_dict['tree_depth']:
            for nu in params_dict['nu']:
                for cascade_depth in params_dict['cascade_depth']:
                    for feature_pool_size in params_dict['feature_pool_size']:
                        for num_test_splits in params_dict['num_test_splits']:
                            for oversampling_amount in params_dict['oversampling_amount']:
                                for oversampling_translation_jitter in params_dict['oversampling_translation_jitter']:
                                    model_save_path = os.path.join(os.getcwd(), 'tmp_model.dat')
                                    model = PupilsDetector(model_save_path, tree_depth, nu, cascade_depth,
                                                           feature_pool_size, num_test_splits, oversampling_amount,
                                                           oversampling_translation_jitter)
                                    model.fit(train_xml_path)
                                    err = model.eval(test_xml_path)
                                    params = f'tree_depth={tree_depth}, nu={nu}, cascade_depth={cascade_depth}, \
                                    feature_pool_size={feature_pool_size}, num_test_splits={num_test_splits}, \
                                    oversampling_amount={oversampling_amount}, \
                                    oversampling_translation_jitter={oversampling_translation_jitter}'
                                    res[params] = err
                                    self.__del_exist_model(model_save_path)
                                    if self.verbose:
                                        n += 1
                                        print(
                                            f'\r[INFO] Estimate {n}/{n_trains} models',
                                            end='',
                                            flush=True
                                        )
        if get_best:
            return self.__get_best_params(res)
        else:
            return res

    def __del_exist_model(self, model_save_path):
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    def __get_best_params(self, grid_search_res):
        best = min(grid_search_res, key=grid_search_res.get)
        best_params = {best: grid_search_res[best]}
        return best_params

    def load(self, model_load_path):
        self.model = dlib.shape_predictor(model_load_path)


if __name__ == '__main__':
    pupils_detector = PupilsDetector(
        model_save_path='../artifacts/pupils_detector.dat',
        tree_depth=3,
        nu=0.1,
        cascade_depth=2,
        feature_pool_size=100,
        num_test_splits=50,
        oversampling_amount=5,
        oversampling_translation_jitter=0.1,
        verbose=True
    )
    pupils_detector.fit('../data/ibug_300W_pupils.xml')
