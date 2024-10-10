import os
from os.path import join as pjoin
import json
from utils import printlog
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Union, Dict, Tuple, List
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import ImageFile, Image
import glob
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OCTID(Dataset):
    random.seed(0)
    valid_modalities = ["OCT", "IR"]
    valid_splits = ["train",  "val", "test", "all", "train_retfound", "val_retfound", "test_retfound"]
    split_to_pathology_to_cnt = {"train": {"NORMAL": 115, "AMRD": 30, "DR": 60, "MH": 60, "CSR": 60},
                                 "val": {"NORMAL": 29, "AMRD": 8, "DR": 15, "MH": 10, "CSR": 10},
                                 "test": {"NORMAL": 62, "AMRD": 17, "DR": 32, "MH": 32, "CSR": 32}}
    # seed 0
    split_to_inds = {"train": [418, 548, 371, 384, 555, 537, 442, 520, 470, 399, 378, 495, 421, 432, 411, 534, 496, 373,
                               403, 518, 446, 477, 497, 571, 546, 504, 522, 479, 527, 455, 550, 560, 491, 554, 458, 382,
                               556, 502, 388, 440, 410, 370, 437, 500, 511, 488, 392, 449, 515, 553, 374, 456, 530, 427,
                               441, 472, 429, 565, 509, 390, 396, 501, 406, 426, 422, 467, 452, 385, 521, 463, 420, 532,
                               417, 505, 507, 478, 462, 473, 512, 476, 545, 380, 393, 474, 552, 492, 412, 568, 567, 528,
                               428, 368, 425, 489, 490, 409, 376, 475, 439, 563, 549, 516, 464, 369, 459, 484, 526, 562,
                               416, 366, 460, 482, 461, 430, 510, 30, 13, 16, 8, 0, 6, 36, 4, 1, 46, 17, 24, 51, 39, 48,
                               15, 49, 50, 54, 29, 33, 53, 44, 9, 7, 26, 52, 31, 41, 14, 211, 163, 214, 241, 222, 252,
                               166, 244, 183, 253, 261, 225, 167, 236, 258, 188, 194, 247, 172, 254, 195, 162, 249, 220,
                               213, 235, 208, 260, 161, 228, 177, 212, 217, 171, 202, 246, 221, 168, 216, 256, 170, 204,
                               164, 207, 176, 189, 180, 234, 242, 184, 255, 160, 209, 181, 230, 250, 199, 173, 231, 262,
                               327, 352, 360, 355, 296, 278, 269, 316, 363, 339, 314, 311, 342, 326, 285, 345, 331, 353,
                               275, 359, 266, 354, 348, 362, 276, 310, 306, 279, 298, 292, 356, 338, 304, 295, 333, 328,
                               274, 346, 336, 361, 364, 265, 267, 325, 281, 300, 283, 264, 273, 305, 341, 340, 357, 315,
                               349, 293, 291, 332, 321, 301, 69, 58, 131, 141, 74, 147, 148, 110, 113, 80, 120, 129, 94,
                               136, 138, 107, 111, 79, 56, 115, 73, 86, 130, 155, 97, 92, 132, 102, 61, 133, 103, 66,
                               65,
                               134, 71, 95, 154, 149, 67, 77, 145, 125, 64, 116, 89, 78, 109, 135, 68, 104, 117, 118,
                               91,
                               70, 119, 75, 57, 90, 72, 82],
                     "val": [407, 435, 415, 414, 451, 569, 379, 519, 389, 557, 503, 386, 381, 444, 466, 536, 570,
                             372, 434, 450, 487, 523, 543, 524, 457, 542, 377, 485, 468, 34, 21, 45, 25, 19, 27, 2, 28,
                             215, 245, 229, 257, 185, 178, 203, 198, 206, 200, 240, 227, 239, 243, 179, 297, 284, 289,
                             318, 294, 350, 268, 288, 287, 320, 84, 121, 59, 96, 100, 105, 112, 114, 99, 87],

                     "test": [401, 423, 431, 367, 486, 551, 408, 471, 498, 544, 383, 404, 499, 419, 525, 494, 400,
                              394, 480, 529, 517, 397, 535, 493, 547, 398, 533, 508, 539, 513, 395, 465, 448, 541,
                              445, 481, 514, 564, 438, 443, 391, 531, 447, 559, 540, 566, 405, 424, 506, 454, 436,
                              453, 402, 387, 375, 469, 561, 433, 558, 483, 413, 538, 37, 10, 20, 40, 11, 32, 43, 23,
                              5, 22, 35, 38, 12, 42, 3, 47, 18, 233, 196, 223, 201, 169, 259, 159, 174, 175, 190,
                              186, 187, 251, 224, 182, 205, 263, 197, 248, 192, 238, 158, 218, 165, 219, 237, 210,
                              226, 191, 232, 157, 193, 280, 324, 330, 335, 270, 343, 286, 271, 347, 272, 303, 302,
                              358, 277, 308, 337, 313, 365, 299, 290, 344, 323, 312, 317, 334, 329, 351, 309, 282,
                              319, 307, 322, 153, 93, 152, 137, 101, 98, 63, 76, 143, 150, 85, 60, 55, 126, 144,
                              142, 140, 122, 146, 139, 127, 62, 156, 124, 108, 128, 106, 151, 81, 123, 83, 88],
                     "all": np.arange(0, 571).tolist(),

                     "train_retfound": ['AMRD1.jpeg', 'AMRD11.jpeg', 'AMRD12.jpeg', 'AMRD13.jpeg',
                                        'AMRD17.jpeg',
                                        'AMRD19.jpeg',
                                        'AMRD2.jpeg',
                                        'AMRD20.jpeg',
                                        'AMRD21.jpeg',
                                        'AMRD22.jpeg',
                                        'AMRD23.jpeg',
                                        'AMRD28.jpeg',
                                        'AMRD29.jpeg',
                                        'AMRD3.jpeg',
                                        'AMRD31.jpeg',
                                        'AMRD37.jpeg',
                                        'AMRD39.jpeg',
                                        'AMRD4.jpeg',
                                        'AMRD41.jpeg',
                                        'AMRD42.jpeg',
                                        'AMRD43.jpeg',
                                        'AMRD44.jpeg',
                                        'AMRD46.jpeg',
                                        'AMRD49.jpeg',
                                        'AMRD5.jpeg',
                                        'AMRD52.jpeg',
                                        'AMRD54.jpeg',
                                        'AMRD6.jpeg',
                                        'AMRD7.jpeg',
                                        'AMRD8.jpeg',
                                        'CSR100.jpeg',
                                        'CSR13.jpeg',
                                        'CSR14.jpeg',
                                        'CSR16.jpeg',
                                        'CSR2.jpeg',
                                        'CSR20.jpeg',
                                        'CSR22.jpeg',
                                        'CSR23.jpeg',
                                        'CSR24.jpeg',
                                        'CSR25.jpeg',
                                        'CSR26.jpeg',
                                        'CSR3.jpeg',
                                        'CSR30.jpeg',
                                        'CSR32.jpeg',
                                        'CSR34.jpeg',
                                        'CSR35.jpeg',
                                        'CSR38.jpeg',
                                        'CSR39.jpeg',
                                        'CSR4.jpeg',
                                        'CSR41.jpeg',
                                        'CSR43.jpeg',
                                        'CSR44.jpeg',
                                        'CSR46.jpeg',
                                        'CSR49.jpeg',
                                        'CSR5.jpeg',
                                        'CSR50.jpeg',
                                        'CSR52.jpeg',
                                        'CSR53.jpeg',
                                        'CSR54.jpeg',
                                        'CSR57.jpeg',
                                        'CSR59.jpeg',
                                        'CSR60.jpeg',
                                        'CSR64.jpeg',
                                        'CSR66.jpeg',
                                        'CSR67.jpeg',
                                        'CSR68.jpeg',
                                        'CSR7.jpeg',
                                        'CSR70.jpeg',
                                        'CSR71.jpeg',
                                        'CSR73.jpeg',
                                        'CSR74.jpeg',
                                        'CSR75.jpeg',
                                        'CSR76.jpeg',
                                        'CSR78.jpeg',
                                        'CSR79.jpeg',
                                        'CSR8.jpeg',
                                        'CSR80.jpeg',
                                        'CSR82.jpeg',
                                        'CSR86.jpeg',
                                        'CSR90.jpeg',
                                        'CSR91.jpeg',
                                        'CSR93.jpeg',
                                        'CSR95.jpeg',
                                        'CSR96.jpeg',
                                        'CSR97.jpeg',
                                        'CSR99.jpeg',
                                        'DR1.jpeg',
                                        'DR101.jpeg',
                                        'DR102.jpeg',
                                        'DR104.jpeg',
                                        'DR106.jpeg',
                                        'DR107.jpeg',
                                        'DR12.jpeg',
                                        'DR13.jpeg',
                                        'DR14.jpeg',
                                        'DR16.jpeg',
                                        'DR17.jpeg',
                                        'DR18.jpeg',
                                        'DR20.jpeg',
                                        'DR21.jpeg',
                                        'DR23.jpeg',
                                        'DR24.jpeg',
                                        'DR27.jpeg',
                                        'DR28.jpeg',
                                        'DR29.jpeg',
                                        'DR33.jpeg',
                                        'DR37.jpeg',
                                        'DR38.jpeg',
                                        'DR41.jpeg',
                                        'DR42.jpeg',
                                        'DR43.jpeg',
                                        'DR44.jpeg',
                                        'DR45.jpeg',
                                        'DR46.jpeg',
                                        'DR47.jpeg',
                                        'DR48.jpeg',
                                        'DR49.jpeg',
                                        'DR5.jpeg',
                                        'DR50.jpeg',
                                        'DR51.jpeg',
                                        'DR53.jpeg',
                                        'DR58.jpeg',
                                        'DR63.jpeg',
                                        'DR64.jpeg',
                                        'DR65.jpeg',
                                        'DR67.jpeg',
                                        'DR68.jpeg',
                                        'DR69.jpeg',
                                        'DR70.jpeg',
                                        'DR71.jpeg',
                                        'DR74.jpeg',
                                        'DR75.jpeg',
                                        'DR76.jpeg',
                                        'DR77.jpeg',
                                        'DR78.jpeg',
                                        'DR79.jpeg',
                                        'DR83.jpeg',
                                        'DR85.jpeg',
                                        'DR86.jpeg',
                                        'DR87.jpeg',
                                        'DR9.jpeg',
                                        'DR91.jpeg',
                                        'DR92.jpeg',
                                        'DR93.jpeg',
                                        'DR96.jpeg',
                                        'MH10.jpeg',
                                        'MH100.jpeg',
                                        'MH101.jpeg',
                                        'MH11.jpeg',
                                        'MH13.jpeg',
                                        'MH14.jpeg',
                                        'MH16.jpeg',
                                        'MH17.jpeg',
                                        'MH18.jpeg',
                                        'MH2.jpeg',
                                        'MH20.jpeg',
                                        'MH23.jpeg',
                                        'MH24.jpeg',
                                        'MH25.jpeg',
                                        'MH26.jpeg',
                                        'MH27.jpeg',
                                        'MH29.jpeg',
                                        'MH3.jpeg',
                                        'MH30.jpeg',
                                        'MH31.jpeg',
                                        'MH32.jpeg',
                                        'MH36.jpeg',
                                        'MH38.jpeg',
                                        'MH39.jpeg',
                                        'MH41.jpeg',
                                        'MH43.jpeg',
                                        'MH44.jpeg',
                                        'MH45.jpeg',
                                        'MH49.jpeg',
                                        'MH50.jpeg',
                                        'MH51.jpeg',
                                        'MH52.jpeg',
                                        'MH53.jpeg',
                                        'MH55.jpeg',
                                        'MH58.jpeg',
                                        'MH59.jpeg',
                                        'MH6.jpeg',
                                        'MH60.jpeg',
                                        'MH62.jpeg',
                                        'MH67.jpeg',
                                        'MH68.jpeg',
                                        'MH70.jpeg',
                                        'MH71.jpeg',
                                        'MH72.jpeg',
                                        'MH73.jpeg',
                                        'MH78.jpeg',
                                        'MH8.jpeg',
                                        'MH80.jpeg',
                                        'MH83.jpeg',
                                        'MH84.jpeg',
                                        'MH85.jpeg',
                                        'MH88.jpeg',
                                        'MH89.jpeg',
                                        'MH94.jpeg',
                                        'MH96.jpeg',
                                        'MH98.jpeg',
                                        'NORMAL10.jpeg',
                                        'NORMAL103.jpeg',
                                        'NORMAL104.jpeg',
                                        'NORMAL105.jpeg',
                                        'NORMAL106.jpeg',
                                        'NORMAL109.jpeg',
                                        'NORMAL11.jpeg',
                                        'NORMAL112.jpeg',
                                        'NORMAL113.jpeg',
                                        'NORMAL114.jpeg',
                                        'NORMAL115.jpeg',
                                        'NORMAL118.jpeg',
                                        'NORMAL119.jpeg',
                                        'NORMAL12.jpeg',
                                        'NORMAL120.jpeg',
                                        'NORMAL121.jpeg',
                                        'NORMAL122.jpeg',
                                        'NORMAL123.jpeg',
                                        'NORMAL125.jpeg',
                                        'NORMAL129.jpeg',
                                        'NORMAL13.jpeg',
                                        'NORMAL130.jpeg',
                                        'NORMAL132.jpeg',
                                        'NORMAL133.jpeg',
                                        'NORMAL134.jpeg',
                                        'NORMAL135.jpeg',
                                        'NORMAL136.jpeg',
                                        'NORMAL137.jpeg',
                                        'NORMAL139.jpeg',
                                        'NORMAL14.jpeg',
                                        'NORMAL140.jpeg',
                                        'NORMAL143.jpeg',
                                        'NORMAL146.jpeg',
                                        'NORMAL149.jpeg',
                                        'NORMAL15.jpeg',
                                        'NORMAL151.jpeg',
                                        'NORMAL153.jpeg',
                                        'NORMAL155.jpeg',
                                        'NORMAL157.jpeg',
                                        'NORMAL158.jpeg',
                                        'NORMAL159.jpeg',
                                        'NORMAL16.jpeg',
                                        'NORMAL160.jpeg',
                                        'NORMAL165.jpeg',
                                        'NORMAL168.jpeg',
                                        'NORMAL169.jpeg',
                                        'NORMAL17.jpeg',
                                        'NORMAL170.jpeg',
                                        'NORMAL171.jpeg',
                                        'NORMAL173.jpeg',
                                        'NORMAL174.jpeg',
                                        'NORMAL175.jpeg',
                                        'NORMAL178.jpeg',
                                        'NORMAL179.jpeg',
                                        'NORMAL18.jpeg',
                                        'NORMAL182.jpeg',
                                        'NORMAL185.jpeg',
                                        'NORMAL186.jpeg',
                                        'NORMAL187.jpeg',
                                        'NORMAL191.jpeg',
                                        'NORMAL196.jpeg',
                                        'NORMAL197.jpeg',
                                        'NORMAL199.jpeg',
                                        'NORMAL20.jpeg',
                                        'NORMAL204.jpeg',
                                        'NORMAL205.jpeg',
                                        'NORMAL206.jpeg',
                                        'NORMAL23.jpeg',
                                        'NORMAL24.jpeg',
                                        'NORMAL25.jpeg',
                                        'NORMAL26.jpeg',
                                        'NORMAL28.jpeg',
                                        'NORMAL30.jpeg',
                                        'NORMAL31.jpeg',
                                        'NORMAL32.jpeg',
                                        'NORMAL35.jpeg',
                                        'NORMAL36.jpeg',
                                        'NORMAL38.jpeg',
                                        'NORMAL39.jpeg',
                                        'NORMAL41.jpeg',
                                        'NORMAL42.jpeg',
                                        'NORMAL44.jpeg',
                                        'NORMAL5.jpeg',
                                        'NORMAL50.jpeg',
                                        'NORMAL52.jpeg',
                                        'NORMAL54.jpeg',
                                        'NORMAL55.jpeg',
                                        'NORMAL57.jpeg',
                                        'NORMAL58.jpeg',
                                        'NORMAL59.jpeg',
                                        'NORMAL60.jpeg',
                                        'NORMAL63.jpeg',
                                        'NORMAL65.jpeg',
                                        'NORMAL66.jpeg',
                                        'NORMAL67.jpeg',
                                        'NORMAL7.jpeg',
                                        'NORMAL70.jpeg',
                                        'NORMAL71.jpeg',
                                        'NORMAL72.jpeg',
                                        'NORMAL75.jpeg',
                                        'NORMAL76.jpeg',
                                        'NORMAL77.jpeg',
                                        'NORMAL78.jpeg',
                                        'NORMAL79.jpeg',
                                        'NORMAL8.jpeg',
                                        'NORMAL82.jpeg',
                                        'NORMAL84.jpeg',
                                        'NORMAL87.jpeg',
                                        'NORMAL88.jpeg',
                                        'NORMAL89.jpeg',
                                        'NORMAL91.jpeg',
                                        'NORMAL92.jpeg',
                                        'NORMAL96.jpeg',
                                        'NORMAL97.jpeg',
                                        'NORMAL98.jpeg'],

                     "val_retfound": ['AMRD15.jpeg','AMRD16.jpeg',
                                        'AMRD24.jpeg',
                                        'AMRD25.jpeg',
                                        'AMRD32.jpeg',
                                        'AMRD35.jpeg',
                                        'AMRD36.jpeg',
                                        'AMRD48.jpeg',
                                        'CSR10.jpeg',
                                        'CSR12.jpeg',
                                        'CSR15.jpeg',
                                        'CSR17.jpeg',
                                        'CSR18.jpeg',
                                        'CSR33.jpeg',
                                        'CSR40.jpeg',
                                        'CSR45.jpeg',
                                        'CSR47.jpeg',
                                        'CSR51.jpeg',
                                        'CSR55.jpeg',
                                        'CSR63.jpeg',
                                        'CSR65.jpeg',
                                        'CSR92.jpeg',
                                        'CSR98.jpeg',
                                        'DR10.jpeg',
                                        'DR100.jpeg',
                                        'DR103.jpeg',
                                        'DR15.jpeg',
                                        'DR39.jpeg',
                                        'DR52.jpeg',
                                        'DR61.jpeg',
                                        'DR66.jpeg',
                                        'DR73.jpeg',
                                        'DR8.jpeg',
                                        'DR80.jpeg',
                                        'DR82.jpeg',
                                        'DR88.jpeg',
                                        'DR94.jpeg',
                                        'DR98.jpeg',
                                        'MH12.jpeg',
                                        'MH19.jpeg',
                                        'MH22.jpeg',
                                        'MH33.jpeg',
                                        'MH48.jpeg',
                                        'MH54.jpeg',
                                        'MH56.jpeg',
                                        'MH63.jpeg',
                                        'MH66.jpeg',
                                        'MH75.jpeg',
                                        'MH79.jpeg',
                                        'MH9.jpeg',
                                        'MH90.jpeg',
                                        'MH95.jpeg',
                                        'MH97.jpeg',
                                        'NORMAL108.jpeg',
                                        'NORMAL116.jpeg',
                                        'NORMAL117.jpeg',
                                        'NORMAL124.jpeg',
                                        'NORMAL127.jpeg',
                                        'NORMAL128.jpeg',
                                        'NORMAL141.jpeg',
                                        'NORMAL145.jpeg',
                                        'NORMAL154.jpeg',
                                        'NORMAL162.jpeg',
                                        'NORMAL176.jpeg',
                                        'NORMAL183.jpeg',
                                        'NORMAL190.jpeg',
                                        'NORMAL192.jpeg',
                                        'NORMAL193.jpeg',
                                        'NORMAL194.jpeg',
                                        'NORMAL201.jpeg',
                                        'NORMAL203.jpeg',
                                        'NORMAL22.jpeg',
                                        'NORMAL3.jpeg',
                                        'NORMAL4.jpeg',
                                        'NORMAL40.jpeg',
                                        'NORMAL46.jpeg',
                                        'NORMAL53.jpeg',
                                        'NORMAL6.jpeg',
                                        'NORMAL62.jpeg',
                                        'NORMAL64.jpeg',
                                        'NORMAL69.jpeg',
                                        'NORMAL80.jpeg'],

                     "test_retfound": ['AMRD10.jpeg',
                                        'AMRD14.jpeg',
                                        'AMRD18.jpeg',
                                        'AMRD26.jpeg',
                                        'AMRD27.jpeg',
                                        'AMRD30.jpeg',
                                        'AMRD33.jpeg',
                                        'AMRD34.jpeg',
                                        'AMRD38.jpeg',
                                        'AMRD40.jpeg',
                                        'AMRD45.jpeg',
                                        'AMRD47.jpeg',
                                        'AMRD50.jpeg',
                                        'AMRD51.jpeg',
                                        'AMRD53.jpeg',
                                        'AMRD55.jpeg',
                                        'AMRD9.jpeg',
                                        'CSR1.jpeg',
                                        'CSR101.jpeg',
                                        'CSR102.jpeg',
                                        'CSR11.jpeg',
                                        'CSR19.jpeg',
                                        'CSR21.jpeg',
                                        'CSR27.jpeg',
                                        'CSR28.jpeg',
                                        'CSR29.jpeg',
                                        'CSR31.jpeg',
                                        'CSR36.jpeg',
                                        'CSR37.jpeg',
                                        'CSR42.jpeg',
                                        'CSR48.jpeg',
                                        'CSR56.jpeg',
                                        'CSR58.jpeg',
                                        'CSR6.jpeg',
                                        'CSR61.jpeg',
                                        'CSR62.jpeg',
                                        'CSR69.jpeg',
                                        'CSR72.jpeg',
                                        'CSR77.jpeg',
                                        'CSR81.jpeg',
                                        'CSR83.jpeg',
                                        'CSR84.jpeg',
                                        'CSR85.jpeg',
                                        'CSR87.jpeg',
                                        'CSR88.jpeg',
                                        'CSR89.jpeg',
                                        'CSR9.jpeg',
                                        'CSR94.jpeg',
                                        'DR105.jpeg',
                                        'DR11.jpeg',
                                        'DR19.jpeg',
                                        'DR2.jpeg',
                                        'DR22.jpeg',
                                        'DR25.jpeg',
                                        'DR26.jpeg',
                                        'DR3.jpeg',
                                        'DR30.jpeg',
                                        'DR31.jpeg',
                                        'DR32.jpeg',
                                        'DR34.jpeg',
                                        'DR35.jpeg',
                                        'DR36.jpeg',
                                        'DR4.jpeg',
                                        'DR40.jpeg',
                                        'DR54.jpeg',
                                        'DR55.jpeg',
                                        'DR56.jpeg',
                                        'DR57.jpeg',
                                        'DR59.jpeg',
                                        'DR6.jpeg',
                                        'DR60.jpeg',
                                        'DR62.jpeg',
                                        'DR7.jpeg',
                                        'DR72.jpeg',
                                        'DR81.jpeg',
                                        'DR84.jpeg',
                                        'DR89.jpeg',
                                        'DR90.jpeg',
                                        'DR95.jpeg',
                                        'DR97.jpeg',
                                        'DR99.jpeg',
                                        'MH1.jpeg',
                                        'MH102.jpeg',
                                        'MH15.jpeg',
                                        'MH21.jpeg',
                                        'MH28.jpeg',
                                        'MH34.jpeg',
                                        'MH35.jpeg',
                                        'MH37.jpeg',
                                        'MH4.jpeg',
                                        'MH40.jpeg',
                                        'MH42.jpeg',
                                        'MH46.jpeg',
                                        'MH47.jpeg',
                                        'MH5.jpeg',
                                        'MH57.jpeg',
                                        'MH61.jpeg',
                                        'MH64.jpeg',
                                        'MH65.jpeg',
                                        'MH69.jpeg',
                                        'MH7.jpeg',
                                        'MH74.jpeg',
                                        'MH76.jpeg',
                                        'MH77.jpeg',
                                        'MH81.jpeg',
                                        'MH82.jpeg',
                                        'MH86.jpeg',
                                        'MH87.jpeg',
                                        'MH91.jpeg',
                                        'MH92.jpeg',
                                        'MH93.jpeg',
                                        'MH99.jpeg',
                                        'NORMAL1.jpeg',
                                        'NORMAL100.jpeg',
                                        'NORMAL101.jpeg',
                                        'NORMAL102.jpeg',
                                        'NORMAL107.jpeg',
                                        'NORMAL110.jpeg',
                                        'NORMAL111.jpeg',
                                        'NORMAL126.jpeg',
                                        'NORMAL131.jpeg',
                                        'NORMAL138.jpeg',
                                        'NORMAL142.jpeg',
                                        'NORMAL144.jpeg',
                                        'NORMAL147.jpeg',
                                        'NORMAL148.jpeg',
                                        'NORMAL150.jpeg',
                                        'NORMAL152.jpeg',
                                        'NORMAL156.jpeg',
                                        'NORMAL161.jpeg',
                                        'NORMAL163.jpeg',
                                        'NORMAL164.jpeg',
                                        'NORMAL166.jpeg',
                                        'NORMAL167.jpeg',
                                        'NORMAL172.jpeg',
                                        'NORMAL177.jpeg',
                                        'NORMAL180.jpeg',
                                        'NORMAL181.jpeg',
                                        'NORMAL184.jpeg',
                                        'NORMAL188.jpeg',
                                        'NORMAL189.jpeg',
                                        'NORMAL19.jpeg',
                                        'NORMAL195.jpeg',
                                        'NORMAL198.jpeg',
                                        'NORMAL2.jpeg',
                                        'NORMAL200.jpeg',
                                        'NORMAL202.jpeg',
                                        'NORMAL21.jpeg',
                                        'NORMAL27.jpeg',
                                        'NORMAL29.jpeg',
                                        'NORMAL33.jpeg',
                                        'NORMAL34.jpeg',
                                        'NORMAL37.jpeg',
                                        'NORMAL43.jpeg',
                                        'NORMAL45.jpeg',
                                        'NORMAL47.jpeg',
                                        'NORMAL48.jpeg',
                                        'NORMAL49.jpeg',
                                        'NORMAL51.jpeg',
                                        'NORMAL56.jpeg',
                                        'NORMAL61.jpeg',
                                        'NORMAL68.jpeg',
                                        'NORMAL73.jpeg',
                                        'NORMAL74.jpeg',
                                        'NORMAL81.jpeg',
                                        'NORMAL83.jpeg',
                                        'NORMAL85.jpeg',
                                        'NORMAL86.jpeg',
                                        'NORMAL9.jpeg',
                                        'NORMAL90.jpeg',
                                        'NORMAL93.jpeg',
                                        'NORMAL94.jpeg',
                                        'NORMAL95.jpeg',
                                        'NORMAL99.jpeg']
                     }

    def __init__(self,
                 root: str,
                 split: str,
                 transforms_dict: Dict[str, Dict[str, List]],
                 img_channels: int = 3,
                 return_metadata: bool = False,
                 debug=False):
        """ Dataset for OCTID dataset """
        self.debug = debug
        self.return_metadata = return_metadata
        # sanity checks
        assert split in self.valid_splits, f'split {split} is not in valid_modes {self.valid_splits}'
        assert os.path.exists(root), f'data path {root} does not exist'

        self.root = root
        self.split = split
        self.modalities = 'OCT'
        self.img_channels = img_channels
        self.img_path = os.path.join(root, 'data')

        # apply split
        if 'retfound' in split:
            # gets paths to images directly
            # remove suffix = '_retfound' from string split
            self.images = [os.path.join(root, 'retfound_splits', split[:-9], p) for p in self.split_to_inds[split]]
        else:
            self.images = glob.glob(os.path.join(self.img_path, '*.jpeg'))
            self.images = [self.images[i] for i in self.split_to_inds[split]]
        self.labels = []

        for path_image in self.images:
            if 'NORMAL' in path_image:
                self.labels.append(0)
            elif 'AMRD' in path_image:
                self.labels.append(1)
            elif 'DR' in path_image:
                self.labels.append(2)
            elif 'MH' in path_image:
                self.labels.append(3)
            elif 'CSR' in path_image:
                self.labels.append(4)
            else:
                raise ValueError(f'Unknown label for image {path_image}')
        # get counts per class
        self.labels = np.array(self.labels)

        self.label_type = 'disease_classification'
        self.label_dict = {'NORMAL': 0.0, 'AMRD': 1.0, 'DR': 2.0, 'MH': 3.0, 'CSR': 4.0}

        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])

        printlog(
            f'OCTID data found \n'
            f'  split = {self.split}, images {self.__len__()} \n '
            f'   \n '
        )
        for p in self.label_dict.keys():
            pathol_count = np.sum(self.labels == self.label_dict[p])
            printlog(f'Number of {p} images: {pathol_count} % {pathol_count / len(self.labels) * 100}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, int]]],
                                          Tuple[torch.Tensor, torch.Tensor]]:
        metadata = {'index': index}
        p = self.images[index]
        label = self.labels[index]

        image = Image.open(p).convert('RGB')
        image, _, metadata = self.common_transforms((image, image, metadata))
        img_tensor, metadata = self.img_transforms((image, metadata))

        if self.debug:
            ToPILImage()(img_tensor).show()
            print(label)

        if self.img_channels == 1:
            img_tensor = img_tensor[0].unsqueeze(0)
            # img_tensor = (1,H,W)

        if self.return_metadata:
            return img_tensor, torch.tensor(label).long(), metadata
        else:
            return img_tensor, torch.tensor(label).long()


if __name__ == '__main__':
    # script to generate train val test and minival.csv
    import json
    from utils import parse_transform_lists
    d = {"dataset": 'OCTID', "experiment": 1}
    data_path = 'C:\\Users\\thopis\\Documents\\datasets\\OCTID\\'
    path_to_config = '../configs/OctBiom/vit_init_both.json'
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']
    transforms_dictionary = {}

    t_dict = parse_transform_lists(transforms_list, transforms_values, **d)

    dataset = OCTID(root=data_path,
                    split='all',
                    transforms_dict=t_dict,
                    debug=True)
    pathologies = ['NORMAL', 'AMRD', 'DR', 'MH', 'CSR']
    for pathology in pathologies:
        pathology_count = np.sum(dataset.labels == dataset.label_dict[pathology])
        print(f'Number of {pathology} images: {pathology_count} % {pathology_count / len(dataset.labels) * 100}')

    np.random.seed(0)  # set seed for reproducibility
    random.seed(0)  # set seed for reproducibility
    for i in [0]:
        # for each split create a list of indices of images from each pathology that are disjoint from that of other splits
        indices = {"train": [], "val": [], "test": []}
        for pathology in pathologies:
            pathology_indices = np.where(dataset.labels == dataset.label_dict[pathology])[0]
            np.random.shuffle(pathology_indices)
            pathology_indices = pathology_indices.tolist()
            cnt = dataset.split_to_pathology_to_cnt["train"][pathology]
            cnt_val = dataset.split_to_pathology_to_cnt["val"][pathology]
            cnt_test = dataset.split_to_pathology_to_cnt["test"][pathology]
            indices["train"] += pathology_indices[:cnt]
            indices["val"] += pathology_indices[cnt:cnt+cnt_val]
            indices["test"] += pathology_indices[cnt+cnt_val:cnt+cnt_val+cnt_test]

        assert len(set(indices['train'] + indices['val'] + indices['test'])) == len(dataset.labels), \
            'some images are in more than one split'

        print(f' iter {i} indices["train"]: {indices["train"]}')
        print(f' iter {i} indices["val"]: {indices["val"]}')
        print(f' iter {i} indices["test"]: {indices["test"]}')
    a = 1
