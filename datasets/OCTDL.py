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
from sklearn.model_selection import train_test_split


# total number of patients 632 (1 ommitted) -> 631
# train
# Number of AMD images: 190 % 43.08390022675737
# Number of DME images: 73 % 16.55328798185941
# Number of ERM images: 41 % 9.297052154195011
# Number of NO images: 66 % 14.965986394557824
# Number of RAO images: 8 % 1.8140589569160999
# Number of RVO images: 32 % 7.2562358276643995
# Number of VID images: 31 % 7.029478458049887


# test
# Number of AMD images: 54 % 42.857142857142854
# Number of DME images: 21 % 16.666666666666664
# Number of ERM images: 12 % 9.523809523809524
# Number of NO images: 19 % 15.079365079365079
# Number of RAO images: 2 % 1.5873015873015872
# Number of RVO images: 9 % 7.142857142857142
# Number of VID images: 9 % 7.142857142857142

# val
# Number of AMD images: 28 % 43.75
# Number of DME images: 11 % 17.1875
# Number of ERM images: 6 % 9.375
# Number of NO images: 9 % 14.0625
# Number of RAO images: 1 % 1.5625
# Number of RVO images: 5 % 7.8125
# Number of VID images: 4 % 6.25

class OCTDL(Dataset):
    random.seed(0)
    valid_modalities = ["OCT", "IR"]
    valid_splits = ["train",  "val", "test", "all"]
    # split_to_pathol_to_patients = {"train": {"NO": 66, "AMD": 190, "DME": 73, "ERM": 41, "RAO": 8, "RVO": 32, "VID": 31},
    #                                "val": {"NO": 9, "AMD": 28, "DME": 11, "ERM": 6, "RAO": 1, "RVO": 5, "VID": 4},
    #                                "test": {"NO": 19, "AMD": 54, "DME": 21, "ERM": 12, "RAO": 2, "RVO": 9, "VID": 9}}

    label_dict = {'NO': 0.0, 'AMD': 1.0, 'DME': 2.0, 'ERM': 3.0, 'RAO': 4.0, 'RVO': 5.0, 'VID': 6.0}
    # seed 0
    split_to_inds = {"train": [585, 586, 587, 588, 364, 365, 277, 278, 279, 280, 578, 579, 1566, 1567, 1579, 104, 115,
                               159, 170, 1230, 1231, 18, 19, 20, 21, 22, 1126, 1128, 1129, 1130, 1131, 437, 438, 1070,
                               1071, 1612, 271, 272, 273, 1146, 867, 868, 869, 870, 1165, 967, 1421, 1168, 1169, 1013,
                               1588, 1589, 1590, 107, 108, 889, 1479, 146, 1366, 1367, 1368, 321, 322, 499, 500, 501,
                               503, 255, 256, 257, 258, 260, 1239, 1240, 1241, 1242, 1243, 426, 427, 428, 429, 430, 248,
                               259, 747, 748, 749, 750, 980, 631, 632, 633, 634, 479, 481, 482, 1282, 751, 752, 1314,
                               1315, 1317, 1318, 1319, 24, 25, 26, 27, 443, 444, 445, 773, 774, 775, 777, 1414, 1415,
                               1416, 592, 593, 403, 404, 1095, 1096, 1097, 1098, 458, 469, 1436, 1437, 1033, 1034, 1035,
                               1508, 1509, 1511, 797, 798, 799, 800, 442, 1153, 1154, 1000, 109, 110, 111, 1027, 1072,
                               1083, 1094, 1105, 1116, 1127, 1138, 1149, 70, 71, 72, 1075, 1076, 172, 173, 174, 1449,
                               778, 779, 780, 781, 782, 391, 402, 753, 755, 605, 606, 607, 608, 609, 610, 611, 612,
                               876, 877, 878, 879, 880, 881, 882, 883, 719, 1385, 1386, 1079, 1080, 637, 638, 245,
                               246, 247, 249, 642, 916, 917, 1545, 1547, 788, 789, 930, 643, 639, 640, 641, 555,
                               556, 557, 996, 56, 67, 1505, 783, 784, 168, 169, 171, 564, 565, 566, 1278, 1279, 1280,
                               1281, 1466, 1477, 1488, 1499, 1510, 1521, 1532, 1533, 141, 142, 1349, 984, 1526, 1527,
                               1570, 1102, 1103, 1104, 899, 1470, 702, 703, 704, 705, 1571, 660, 661, 1512, 1513, 1514,
                               270, 1049, 1051, 1144, 649, 650, 651, 652, 653, 655, 1199, 1200, 303, 413, 1535, 1546,
                               1552, 1553, 567, 568, 570, 447, 807, 818, 998, 155, 156, 157, 765, 776, 1155, 521, 522,
                               523, 720, 722, 723, 724, 83, 84, 85, 86, 353, 354, 355, 175, 176, 506, 507, 508, 509,
                               510, 920, 1419, 1125, 1020, 1530, 843, 844, 1043, 1337, 1338, 1426, 1427, 1428, 1613,
                               1614, 810, 811, 812, 813, 814, 815, 816, 817, 1569, 891, 892, 1166, 1167, 121, 122,
                               274, 275, 276, 263, 264, 265, 266, 267, 268, 269, 1447, 1448, 77, 79, 912, 316, 317,
                               318, 319, 320, 1141, 943, 1422, 1073, 1074, 968, 1531, 1381, 1382, 574, 575, 576, 577,
                               851, 1610, 1611, 1417, 1418, 808, 809, 149, 150, 151, 1204, 1206, 1207, 1373, 1374, 1375,
                               1376, 1377, 328, 329, 330, 331, 1482, 1483, 117, 1162, 1412, 559, 560, 561, 1459, 1460,
                               1461, 1462, 841, 842, 28, 29, 30, 1092, 1093, 386, 387, 388, 389, 390, 78, 89, 1215,
                               1217, 1218, 1219, 1220, 1221, 1222, 1534, 871, 872, 874, 875, 733, 734, 735, 736, 737,
                               738, 1212, 1213, 1214, 490, 492, 493, 494, 940, 941, 1068, 1069, 589, 590, 1399, 1400,
                               1401, 1403, 468, 470, 471, 472, 473, 474, 75, 76, 1592, 1593, 300, 301, 302, 464, 465,
                               466, 467, 1413, 1424, 1587, 1229, 1004, 1170, 1171, 1173, 1195, 1196, 1197, 1198, 1006,
                               411, 412, 415, 416, 58, 59, 60, 61, 534, 535, 957, 304, 315, 893, 894, 896, 0, 480, 491,
                               502, 513, 525, 536, 992, 1106, 140, 87, 88, 90, 91, 92, 93, 94, 95, 96, 983, 1001, 991,
                               1052, 1053, 281, 292, 483, 484, 485, 691, 692, 693, 1232, 1233, 1503, 1609, 1464, 73, 74, 125, 127, 982, 1123, 1124, 969, 1536, 1537, 1556, 1557, 1558, 1161, 1172, 1183, 1402, 971, 1201, 1202, 1203, 99, 101, 102, 103, 1457, 1458, 935, 936, 937, 382, 383, 261, 262, 407, 408, 409, 410, 1467, 1468, 80, 81, 82, 1208, 1209, 1210, 1192, 1193, 1056, 1057, 1058, 123, 124, 1002, 1562, 1563, 885, 999, 6, 7, 44, 46, 47, 48, 981, 230, 231, 1336, 1347, 1358, 1369, 1380, 1391, 475, 476, 477, 478, 835, 836, 913, 914, 942, 1598, 740, 741, 742, 744, 745, 746, 1086, 1087, 1088, 1089, 1431, 1432, 1433, 1395, 1396, 1397, 1398, 414, 425, 436, 849, 850, 1180, 1181, 1182, 1184, 1185, 1186, 1572, 100, 105, 1478, 1077, 1078, 677, 678, 679, 680, 1615, 829, 840, 1158, 1159, 1107, 1211, 332, 333, 334, 335, 224, 225, 227, 228, 229, 8, 9, 10, 11, 13, 422, 423, 424, 1309, 1310, 1311, 1312, 1313, 966, 977, 946, 1440, 1441, 1442, 1443, 1580, 1, 181, 993, 939, 954, 1450, 1451, 1452, 1453, 1454, 1456, 1473, 1474, 732, 743, 754, 790, 791, 792, 793, 953, 1014, 619, 620, 621, 622, 623, 1258, 1259, 1261, 526, 527, 528, 529, 530, 531, 532, 533, 1480, 1187, 1188, 886, 1009, 1491, 504, 505, 1529, 1143, 1084, 1085, 1363, 1364, 446, 448, 449, 450, 144, 145, 1425, 787, 794, 795, 392, 393, 4, 5, 819, 820, 821, 822, 823, 824, 825, 826, 1133, 1134, 1135, 1136, 1137, 1139, 1140, 307, 308, 1434, 975, 976, 1549, 1608, 120, 305, 306, 152, 153, 154, 143, 923, 924, 1586, 232, 233, 234, 235, 1600, 950, 951, 1484, 1485, 1486, 1487, 1489, 1490, 927, 1046, 1047, 1048, 12, 23, 1016, 511, 512, 1573, 685, 686, 688, 689, 624, 636, 645, 646, 647, 648, 253, 254, 1559, 1550, 985, 756, 757, 758, 759, 282, 283, 284, 285, 286, 287, 697, 699, 700, 701, 934, 1469, 1054, 1055, 158, 160, 161, 162, 845, 846, 847, 848, 1244, 1245, 1246, 1247, 1248, 1250, 997, 1564, 1297, 1256, 1257, 1003, 1234, 1235, 1236, 1237, 1465, 106, 1148, 1150, 1151, 884, 922, 1194, 1205, 1216, 1227, 962, 431, 432, 433, 434, 990, 163, 164, 165, 1577, 1578, 656, 657, 658, 659, 907, 915, 1145, 1189, 1190, 1191, 918, 919, 1099, 1100, 1101, 1108, 288, 289, 290, 291, 293, 294, 186, 187, 188, 189, 190, 1174, 1504, 1471, 1472, 547, 558, 569, 580, 591, 602, 613, 1290, 1291, 1292, 1293, 1295, 1296, 1010, 1019, 1599, 785, 786, 978, 979, 947, 183, 184, 185, 451, 452, 453, 454, 455, 456, 193, 204, 207, 1270, 1273, 827, 828, 830, 831, 890, 1163, 1164, 1492, 988, 359, 360, 361, 362, 363, 898, 16, 17, 1040, 1041, 1042, 62, 1565, 711, 712, 713, 714, 715, 1476, 972, 973, 974, 417, 418, 419, 420, 421, 864, 865, 866, 1506, 1507, 400, 401, 1585, 571, 572, 573, 309, 310, 311, 312, 313, 314, 1582, 1387, 1388, 1389, 956, 928, 1602, 1603, 933, 1323, 1324, 1325, 1326, 1328, 1329, 1330, 1331, 435, 524, 635, 654],

                     "val":  [989, 1524, 130, 112, 113, 1156, 1584, 1152, 627, 628, 629, 630, 709, 796, 1515, 1516, 374,
                              375, 376, 377, 1322, 49, 50, 51, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1306, 1132,
                              1015, 667, 668, 1350, 1351, 1352, 1114, 1115, 1117, 1118, 1119, 614, 615, 616, 617, 618,
                              133, 1423, 1029, 961, 36, 37, 38, 39, 1147, 1307, 1308, 1605, 1268, 1269, 1554, 1555, 97,
                              98, 1274, 1275, 986, 987, 495, 496, 497, 498, 837, 838, 839, 1334, 1335, 716, 717, 718,
                              1541, 1542, 1544, 1011, 1026, 859, 860, 861, 863, 31, 32, 33, 35, 546, 548, 549, 550,
                              671, 672, 673, 674, 675, 132, 1604, 236, 238, 239, 1560, 1561, 895, 909, 910, 486, 487,
                              488, 489, 1444, 1455, 929, 960, 134, 135, 65, 66, 68, 69, 958, 1551, 208, 994, 995, 240,
                              241, 242, 243, 244, 212, 213, 214, 216, 1365, 694, 695, 696, 1284, 1285, 1286, 1287, 1288,
                              1289, 40, 41, 42, 43],


                     "test": [1594, 551, 552, 553, 554, 963, 1481, 965, 767, 768, 769, 770, 771, 772, 1294, 1305, 1316,
                              1327, 1332, 1333, 970, 147, 366, 367, 368, 897, 925, 926, 1050, 1061, 1066, 1067, 1036,
                              1037, 1038, 739, 1435, 1438, 1439, 1463, 138, 669, 670, 945, 1030, 1031, 1032, 370, 371,
                              372, 373, 223, 1404, 1405, 944, 955, 1044, 1045, 1223, 1224, 1225, 1226, 1228, 1160, 725,
                              726, 727, 728, 729, 730, 731, 136, 1445, 14, 15, 931, 932, 902, 903, 1378, 1379, 854, 855,
                              856, 857, 858, 801, 802, 803, 804, 518, 519, 520, 964, 1142, 888, 1519, 1520, 1522, 1523,
                              166, 167, 1059, 1028, 1039, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1348, 1429,
                              1430, 938, 457, 459, 460, 461, 462, 463, 537, 538, 539, 540, 959, 325, 336, 347, 358, 369,
                              380, 594, 595, 596, 597, 598, 599, 600, 601, 603, 604, 1596, 1597, 139, 1540, 1543, 1359,
                              1007, 1008, 118, 119, 1607, 1538, 1539, 194, 195, 196, 197, 2, 3, 191, 192, 215, 226, 237,
                              1120, 1121, 1122, 45, 34, 1370, 1371, 1372, 1112, 1113, 1583, 323, 324, 326, 327, 1109,
                              1110, 1111, 217, 218, 219, 220, 221, 222, 805, 806, 1494, 1495, 1005, 1517, 1518, 1081,
                              1082, 541, 542, 543, 544, 545, 1021, 1022, 1023, 1574, 1606, 250, 251, 252, 126, 137, 148,
                              1175, 1176, 1177, 1178, 1179, 63, 64, 1320, 1321, 1157, 1390, 1392, 1393, 1394, 114, 116,
                              1060, 1062, 1406, 1407, 1408, 1409, 1410, 1411, 131, 1276, 1277, 1493, 1238, 1249, 1260,
                              1272, 1283, 1475, 1420, 198, 199, 200, 201, 1525, 1528, 583, 584, 1595, 1601, 706, 707,
                              708, 949, 852, 853, 832, 833, 834, 662, 663, 664, 666, 1063, 1064, 1065, 562, 563, 295,
                              296, 297, 298, 299, 1360, 1361, 1362, 128, 129, 625, 626, 760, 761, 762, 763, 405, 406,
                              1090, 1091, 394, 395, 396, 948, 397, 398, 399, 1591, 581, 582, 1353, 1354, 1355, 1356,
                              1357, 202, 203, 205, 206, 921, 764, 766, 337, 338, 339, 378, 379, 381, 710, 721, 862, 873,
                              1446, 52, 53, 54, 55, 57, 1581, 644, 1568, 887, 1575, 1576, 1616, 900, 901, 439, 440, 441,
                              904, 905, 906, 908, 177, 178, 179, 180, 182, 1548, 209, 210, 211, 1271, 1496, 1497, 1498,
                              1500, 1501, 1502, 1262, 1263, 1264, 1265, 1266, 1267, 348, 349, 350, 351, 352, 1251, 1252,
                              1253, 1254, 1255, 1024, 1025, 384, 385, 1012, 356, 357, 665, 676, 687, 698, 681, 682, 683,
                              684, 514, 515, 516, 517, 1383, 1384, 340, 341, 342, 343, 344, 345, 346, 952, 1017, 1018],

                     "all": np.arange(0, 1617).tolist()  # used to be 1617 because one patient has two labels
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
        self.img_path = root
        self.split = split
        self.modalities = 'OCT'
        self.img_channels = img_channels
        meta_file = Path(root) / 'OCTDL_labels.csv'
        # open metadata file
        metadata = pd.read_csv(meta_file)
        # get columns 'file_name' and 'disease'
        metadata = metadata[['file_name', 'disease', 'patient_id']].values.tolist()
        self.diseases = [x[1] for x in metadata]
        self.patient_ids = [x[2]-1 for x in metadata]

        # set label type and label dictionary
        self.label_type = 'disease_classification'
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}

        # get occurences of each patient_id and diseases for each patient_id
        self.unique_patient_ids = list(set(self.patient_ids))
        self.unique_diseases = list(set(self.diseases))
        # map patient to image indices
        self.patient_id_to_inds = {pid: np.where(np.array(self.patient_ids) == pid)[0].tolist() for pid in self.unique_patient_ids}
        # map disease to image indices
        self.disease_to_inds = {disease: np.where(np.array(self.diseases) == disease)[0].tolist() for disease in self.unique_diseases}
        # map patient to diseases
        self.patient_id_to_diseases = {pid: list(set(np.array(self.diseases)[self.patient_id_to_inds[pid]].tolist()))
                                       for pid in self.unique_patient_ids}

        # find patient that has more than one disease
        # we keep only patients with exactly one disease associated with their images

        patient_ids_with_more_than_one_disease = [pid for pid, diseases in self.patient_id_to_diseases.items()
                                                  if len(diseases) > 1]
        printlog(f'*** patient_ids with more than one disease: {patient_ids_with_more_than_one_disease} ')
        if len(patient_ids_with_more_than_one_disease) > 0:
            print(f'patient_ids with more than one disease: {patient_ids_with_more_than_one_disease}')
            # remove images from patients with more than one disease
            for pid in patient_ids_with_more_than_one_disease:
                inds_to_remove = self.patient_id_to_inds[pid]
                printlog(f"removing images with inds: {inds_to_remove} for patient {pid}")
                self.patient_id_to_diseases.pop(pid)
                self.unique_patient_ids.remove(pid)
                self.patient_id_to_inds.pop(pid)
                self.diseases = [x for i, x in enumerate(self.diseases) if i not in inds_to_remove]
                # for x in metadata:
                #     if x[2] == pid:
                #         metadata.remove(x)

        self.patiend_id_to_labels = {pid: self.label_dict[disease[0]] for pid, disease in self.patient_id_to_diseases.items()}

        # get all images
        all_images = [os.path.join(self.img_path, x[1], x[0]) for x in metadata]
        # apply split
        self.images = [] # images used for this split
        for i in self.split_to_inds[split]:
            self.images.append(all_images[i])

        self.labels = []  # to store labels per image

        for path_image in self.images:
            if 'norm' in path_image:
                self.labels.append(0)
            elif 'amd' in path_image:
                self.labels.append(1)
            elif 'dme' in path_image:
                self.labels.append(2)
            elif 'erm' in path_image:
                self.labels.append(3)
            elif 'rao' in path_image:
                self.labels.append(4)
            elif 'rvo' in path_image:
                self.labels.append(5)
            elif 'vid' in path_image:
                self.labels.append(6)
            else:
                raise ValueError(f'Unknown label for image {path_image}')

        # get counts per class
        self.labels = np.array(self.labels)
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])

        printlog(
            f'OCTDL data found \n'
            f'  split = {self.split}, images {self.__len__()} \n '
            f'   \n '
        )
        for p in self.label_dict.keys():
            pathol_count = np.sum(self.labels == self.label_dict[p])
            printlog(f'Number of {p} images: {pathol_count} % {pathol_count / len(self.labels) * 100}')

    def get_integer_label(self, label: str) -> int:
        return self.label_dict[label]


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
            print(self.inv_label_dict[label])

        if self.img_channels == 1:
            img_tensor = img_tensor[0].unsqueeze(0)
            # img_tensor = (1,H,W)

        if self.return_metadata:
            return img_tensor, torch.tensor(label).long(), metadata
        else:
            return img_tensor, torch.tensor(label).long()


