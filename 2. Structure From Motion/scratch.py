from common.dataset import Dataset
from common.trajectory import Trajectory
from common.intrinsics import Intrinsics

import os
import math
import pickle
import cv2 as cv
import numpy as np
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict, Counter
from matplotlib import pyplot as plt

def serialize_keypoints(keypoints):
    keypoints_as_list = []
    for kp in keypoints:
        temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        keypoints_as_list.append(temp)
    return keypoints_as_list

def deserialize_keypoints(array):
    keypoints = []
    for point, size, angle, response, octave, class_id in array:
        kp = cv.KeyPoint(point[0], point[1], size, angle, response, octave, class_id)
        keypoints.append(kp)
    return keypoints

def computeORB_inmemory(data_dir):
    """
    Вычисляет ORB признаки для всех изображений из rgb.txt и возвращает словарь,
    где ключ — номер изображения, а значение — кортеж (descriptors, keypoints).
    Также возвращается словарь путей из rgb.txt.
    """
    rgb_path = Dataset.get_rgb_list_file(data_dir)
    assert os.path.isfile(rgb_path), "rgb.txt не найден в data_dir"
    test_name = os.path.basename(os.path.normpath(data_dir))
    path_list = Dataset.read_dict_of_lists(rgb_path, key_type=int)
    features = {}
    for num, path in tqdm(path_list.items(), desc="Вычисление ORB признаков"):
        img_path = os.path.join(data_dir, path)
        img_bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            continue
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        orb = cv.ORB_create(nfeatures=900)
        kp = orb.detect(img_rgb, None)
        kp, des = orb.compute(img_rgb, kp)
        features[num] = (des, kp)
    return features, path_list

def computeInliers_inmemory(features):
    """
    Выполняет сопоставление дескрипторов между всеми парами изображений из features.
    Возвращает словарь, где ключ — пара (img_id1, img_id2), а значение — список инлайер-совпадений.
    """
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    inlier_matches = {}
    sorted_keys = sorted(features.keys())
    for i in tqdm(range(len(sorted_keys)), desc="Обработка пар изображений"):
        id_i = sorted_keys[i]
        des_i, kp_i = features[id_i]
        for j in range(i + 1, len(sorted_keys)):
            id_j = sorted_keys[j]
            des_j, kp_j = features[id_j]
            matches = bf.knnMatch(des_i, des_j, k=2)
            ratio_thresh = 0.3
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            if len(good_matches) >= 8:
                pts_i = np.float32([kp_i[m.queryIdx].pt for m in good_matches])
                pts_j = np.float32([kp_j[m.trainIdx].pt for m in good_matches])
                F, mask = cv.findFundamentalMat(pts_i, pts_j, cv.FM_RANSAC, 3.0, 0.99)
                if mask is not None:
                    inliers = [good_matches[k] for k in range(len(good_matches)) if mask[k]]
                else:
                    inliers = []
                inlier_matches[(id_i, id_j)] = inliers
            else:
                inlier_matches[(id_i, id_j)] = []
    return inlier_matches


'''def computeInliers_inmemory(features):
    """
    Выполняет сопоставление дескрипторов между всеми парами изображений из features с использованием FLANN-матчера для бинарных дескрипторов.
    Возвращает словарь, где ключ — пара (img_id1, img_id2), а значение — список инлайер-совпадений.
    
    Аргументы:
      features: словарь, где ключ — номер изображения, а значение — кортеж (descriptors, keypoints)
                (например, полученный из функции computeORB_inmemory).
    """
    # Настройка FLANN для бинарных дескрипторов (LSH)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,      # можно экспериментировать, 6-12
                        key_size=12,         # обычно 12-20
                        multi_probe_level=1) # уровень мульти-запроса
    search_params = dict(checks=50)  # число проверок
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    inlier_matches = {}
    sorted_keys = sorted(features.keys())
    
    for i in tqdm(range(len(sorted_keys)), desc="Обработка пар изображений"):
        id_i = sorted_keys[i]
        des_i, kp_i = features[id_i]
        # Убедимся, что дескрипторы имеют тип np.uint8
        des_i = np.asarray(des_i, dtype=np.uint8)
        for j in range(i + 1, len(sorted_keys)):
            id_j = sorted_keys[j]
            des_j, kp_j = features[id_j]
            des_j = np.asarray(des_j, dtype=np.uint8)
            try:
                matches = flann.knnMatch(des_i, des_j, k=2)
            except Exception as e:
                print(f"Ошибка FLANN для пары ({id_i}, {id_j}): {e}")
                continue

            ratio_thresh = 0.75
            good_matches = []
            for match in matches:
                if len(match) < 2:
                    continue  # пропускаем, если соседей меньше 2
                m, n = match[0], match[1]
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            if len(good_matches) >= 8:
                pts_i = np.float32([kp_i[m.queryIdx].pt for m in good_matches])
                pts_j = np.float32([kp_j[m.trainIdx].pt for m in good_matches])
                F, mask = cv.findFundamentalMat(pts_i, pts_j, cv.FM_RANSAC, 3.0, 0.99)
                if mask is not None:
                    inliers = [good_matches[k] for k in range(len(good_matches)) if mask[k]]
                else:
                    inliers = []
                inlier_matches[(id_i, id_j)] = inliers
            else:
                inlier_matches[(id_i, id_j)] = []
    return inlier_matches'''



def filterInliers(inlier_matches, verbose=False):
    """
    Объединяет наблюдения из inlier_matches в треки с помощью Union-Find и отфильтровывает треки,
    в которых для одного изображения обнаружено более одной наблюдаемой точки (т.е. конфликт).
    
    Аргументы:
      inlier_matches: словарь, где ключ – пара (img_id1, img_id2), а значение – список объектов cv.DMatch.
      verbose: если True, выводит статистику по трекам.
      
    Возвращает:
      valid_tracks: словарь, где ключ – идентификатор трека, а значение – отсортированный список наблюдений
                     (каждое наблюдение представлено как (img_id, kp_index)).
    """
    from collections import defaultdict, Counter
    from tqdm import tqdm

    class UnionFind:
        def __init__(self):
            self.parent = {}
        
        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                return x
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            rootX = self.find(x)
            rootY = self.find(y)
            if rootX != rootY:
                self.parent[rootY] = rootX

    uf = UnionFind()
    
    # Объединяем наблюдения из всех пар inlier_matches
    for (img1, img2), matches in tqdm(inlier_matches.items(), desc="Объединяем наблюдения"):
        for m in matches:
            obs1 = (img1, m.queryIdx)
            obs2 = (img2, m.trainIdx)
            uf.union(obs1, obs2)
    
    # Группируем наблюдения по их корневым элементам
    tracks = defaultdict(list)
    for obs in uf.parent.keys():
        root = uf.find(obs)
        tracks[root].append(obs)
    
    # Отфильтровываем треки: если в треке для одного изображения более одной наблюдаемой точки или трек содержит менее 2 наблюдений, считаем его недостоверным
    valid_tracks = {}
    bad_tracks = {}
    for track_id, obs_list in tracks.items():
        img_occurrences = defaultdict(int)
        conflict = False
        for obs in obs_list:
            img, _ = obs
            img_occurrences[img] += 1
            if img_occurrences[img] > 1:
                conflict = True
                break
        if conflict or len(obs_list) < 2:
            bad_tracks[track_id] = obs_list
        else:
            valid_tracks[track_id] = sorted(obs_list, key=lambda pair: pair[0])
    
    if verbose:
        total_points = sum(len(v) for v in tracks.values())
        bad_points = sum(len(v) for v in bad_tracks.values())
        print(f'Отфильтровано {bad_points} наблюдений из {total_points} ({bad_points/total_points:.2f} от общего числа).')
        print(f'Всего треков: {len(tracks)}, хороших: {len(valid_tracks)}.')
        from collections import Counter
        good_tracks_stats = Counter(len(v) for v in valid_tracks.values())
        sorted_good = sorted(good_tracks_stats.items(), key=lambda x: x[0])
        print("Распределение длин хороших треков (по возрастанию):", sorted_good)
        bad_tracks_stats = Counter(len(v) for v in bad_tracks.values())
        sorted_bad = sorted(bad_tracks_stats.items(), key=lambda x: x[0])
        print("Распределение длин плохих треков (по возрастанию):", sorted_bad)
    
    return valid_tracks


# Пример использования:
if __name__ == '__main__':
    data_dir = './tests/00_test_slam_input'
    # Вычисляем ORB признаки для всех изображений (in-memory)
    features, path_list = computeORB_inmemory(data_dir)
    # Выполняем сопоставление и находим инлайеры между всеми парами
    inlier_matches = computeInliers_inmemory(features)
    valid_tracks = filterInliers(inlier_matches, verbose=True)
