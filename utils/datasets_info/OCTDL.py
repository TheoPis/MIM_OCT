

# todo - add the following to the dataset info:
from typing import Any


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

#  NPDR vs PDR


classes_exp0 = {
    0: "NO",  # control
    1: "AMD",    # age-related macular degeneration
    2: "DME",    # diabetic retinopathy
    3: "ERM",    # epiretinal membrane
    4: "RAO",    # retinal artery occlusion
    5: "RVO",    # retinal vein occlusion
    6: "VID",    # Vitreomacular Interface Disease
}

class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6]
}


categories_exp0 = {
    "diseases": [1, 2, 3, 4, 5, 6],
    "healthy": [0]
}


classes_exp1 = {
    0: "NO",  # control
    1: "AMD",  # age-related macular degeneration
    2: "DME",  # diabetic retinopathy
    3: "ERM",  # epiretinal membrane
    4: "RAO",  # retinal artery occlusion
    5: "RVO",  # retinal vein occlusion
    6: "VID",  # Vitreomacular Interface Disease
}

class_remapping_exp1 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6]
}

categories_exp1 = {
    "diseases": [1, 2, 3, 4, 5, 6],
    "healthy": [0]
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]]

NUM_CLASSES = [len(CLASS_INFO[0][1]), len(CLASS_INFO[1][1])]


OCTDL_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)


def label_sanity_check(root=None):
    #   split = train, images 903
    #
    #
    # Number of NO images: 149 % 16.500553709856035
    # Number of AMD images: 505 % 55.924695459579176
    # Number of DME images: 73 % 8.084163898117387
    # Number of ERM images: 73 % 8.084163898117387
    # Number of RAO images: 12 % 1.3289036544850499
    # Number of RVO images: 63 % 6.976744186046512
    # Number of VID images: 28 % 3.10077519379845

    #   split = val, images 360
    #
    #
    # Number of NO images: 70 % 19.444444444444446
    # Number of AMD images: 190 % 52.77777777777778
    # Number of DME images: 35 % 9.722222222222223
    # Number of ERM images: 30 % 8.333333333333332
    # Number of RAO images: 5 % 1.3888888888888888
    # Number of RVO images: 15 % 4.166666666666666
    # Number of VID images: 15 % 4.166666666666666

    #   split = test, images 355
    #
    #
    # Number of NO images: 65 % 18.30985915492958
    # Number of AMD images: 190 % 53.52112676056338
    # Number of DME images: 35 % 9.859154929577464
    # Number of ERM images: 30 % 8.450704225352112
    # Number of RAO images: 5 % 1.4084507042253522
    # Number of RVO images: 15 % 4.225352112676056
    # Number of VID images: 15 % 4.225352112676056

    pass


if __name__ == '__main__':
    label_sanity_check()
