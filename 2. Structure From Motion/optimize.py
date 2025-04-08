import os
import optuna
import subprocess
import numpy as np
import tempfile
import shutil
import time
import json
from types import SimpleNamespace

from common.dataset import Dataset
from common import testing
from common.trajectory import Trajectory

def objective(trial):
    """
    Функция цели для оптимизации.
    Гиперпараметры передаются через params в estimate_trajectory.
    После запуска алгоритма вызывается "python3 run.py tests" для проверки решения по всем тестам,
    после чего извлекается средняя оценка (Mark).
    """
    # Определяем гиперпараметры, передаваемые через params
    params = SimpleNamespace(
        ORBCount = trial.suggest_int('ORBCount', 500, 2500),
        ransac_reproj_threshold = trial.suggest_float('ransac_reproj_threshold', 2.0, 5.0),
        confidence_level = trial.suggest_float('confidence_level', 0.985, 0.995),
        lowe_ratio_threshold = trial.suggest_float('lowe_ratio_threshold', 0.3, 0.8),
        reproj_threshold = trial.suggest_float('reproj_threshold', 5.0, 15.0)
    )
    
    # Здесь data_dir не используется напрямую, так как run.py в локальном режиме ожидает только tests_dir.
    # Мы генерируем код с параметрами (например, в estimate_trajectory.py) через функцию generate_code_with_params.
    # Для простоты здесь предполагается, что ваш скрипт оптимизации перезаписывает estimate_trajectory.py с нужными параметрами.
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_filename = temp_file.name
    try:
        generate_code_with_params(temp_filename, params)
        # Копируем сгенерированный файл в estimate_trajectory.py
        shutil.copy(temp_filename, 'estimate_trajectory.py')
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    # Запускаем run.py в локальном режиме, чтобы проверить решение по всем тестам
    tests_dir = "./tests"  # верхняя директория с тестами
    try:
        result = subprocess.run(
            ["python3", "run.py", tests_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        output = result.stdout
        # Ищем строку вида "Mark: X.XX" в выводе
        mark = 0.0
        for line in output.splitlines():
            if line.startswith("Mark:"):
                # Например, строка выглядит как "Mark: 9.86 tr 0.01 rot 0.35"
                try:
                    mark = float(line.split()[1])
                except Exception as e:
                    print("Ошибка парсинга оценки:", e)
                    mark = 0.0
                break
        print(f"Trial finished with mark = {mark:.2f} and parameters: {vars(params)}")
    except Exception as e:
        print("Ошибка в objective:", e)
        mark = 0.0

    return mark

def generate_code_with_params(filename, opt_params):
    """
    Генерирует файл с кодом функции estimate_trajectory с заданными параметрами.
    Вставляем параметры в виде констант внутри кода.
    """
    code = f"""
from common.dataset import Dataset
from common.trajectory import Trajectory
from common.intrinsics import Intrinsics

import os
import glob
import math
import pickle
import cv2 as cv
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from itertools import combinations
from matplotlib import pyplot as plt
from collections import defaultdict, Counter

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


def serialize_dmatches(matches):
        return [
            {{"queryIdx": m.queryIdx, "trainIdx": m.trainIdx, "imgIdx": m.imgIdx, "distance": m.distance}}
            for m in matches
        ]


def deserialize_dmatches(serialized_matches):
    matches = []
    for d in serialized_matches:
        m = cv.DMatch(d["queryIdx"], d["trainIdx"], d["imgIdx"], d["distance"])
        matches.append(m)
    return matches


def computeORB(data_dir, features_dir, params, reCompute=True):
    rgb_path = Dataset.get_rgb_list_file(data_dir)
    assert os.path.isfile(rgb_path), "rgb.txt не найден в data_dir"
    test_name = data_dir.split('/')[-1]
    print(test_name)

    keypoints = {{}}
    descriptors = {{}}

    path_list = Dataset.read_dict_of_lists(rgb_path, key_type=int)
    for frame_id, path in tqdm(path_list.items(), desc="Вычисление дескрипторов"):
        des_path = os.path.join(features_dir, f'{{test_name}}_{{frame_id}}_des.pkl')
        kp_path = os.path.join(features_dir, f'{{test_name}}_{{frame_id}}_kp.pkl')
        img_path = os.path.join(data_dir, path)

        if not reCompute and os.path.exists(des_path) and os.path.exists(kp_path):
            with open(des_path, 'rb') as f:
                descriptors[frame_id] = pickle.load(f)
            with open(kp_path, 'rb') as f:
                keypoints_serialised = pickle.load(f)
                keypoints[frame_id] = deserialize_keypoints(keypoints_serialised)
        else:
            img_bgr = cv.imread(img_path, cv.IMREAD_COLOR)
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

            orb = cv.ORB_create(nfeatures=params.ORBCount)
            kp = orb.detect(img_rgb, None)
            kp, des = orb.compute(img_rgb, kp)

            descriptors[frame_id] = des
            keypoints[frame_id] = kp

            #img_upd = cv.drawKeypoints(img_rgb, kp, None, color=(0,255,0), flags=0)
            #plt.imshow(img_upd)
            #plt.show()
        
            with open(des_path, 'wb') as f:
                pickle.dump(des, f)

            keypoints_serialised = serialize_keypoints(kp)
            with open(kp_path, 'wb') as f:
                pickle.dump(keypoints_serialised, f)
    return keypoints, descriptors


def computeInliers(keypoints, descriptors, known_frames, params, data_dir, features_dir='./data', verbose=0, reCompute=True):
    test_name = data_dir.split('/')[-1]
    save_file = os.path.join(features_dir, f'{{test_name}}_inlier_matches.pkl')

    if not reCompute and os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            inlier_matches = pickle.load(f)
    else:
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        
        inlier_matches = {{}}
        
        for i in tqdm(range(len(known_frames)), desc="Матчинг инлайеров"):
            ith_frame_id = known_frames[i]
            des_i = descriptors[ith_frame_id]
            kp_i = keypoints[ith_frame_id]
            
            for j in range(i + 1, len(known_frames)):
                jth_frame_id = known_frames[j]
                des_j = descriptors[jth_frame_id]
                kp_j = keypoints[jth_frame_id]
                
                matches = bf.knnMatch(des_i, des_j, k=2)

                ratio_thresh = params.lowe_ratio_threshold
                good_matches = []
                for m, n in matches:
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) >= 8:
                    pts_i = np.float32([kp_i[m.queryIdx].pt for m in good_matches])
                    pts_j = np.float32([kp_j[m.trainIdx].pt for m in good_matches])

                    F, mask = cv.findFundamentalMat(pts_i, pts_j, cv.FM_RANSAC, params.ransac_reproj_threshold, params.confidence_level)
                    if mask is not None:
                        inliers = []
                        for k, match in enumerate(good_matches):
                            if mask[k][0]:
                                inliers.append((match.queryIdx, match.trainIdx))

                    inlier_matches[(ith_frame_id, jth_frame_id)] = inliers
                    if verbose == 2:
                        print(f"Пара ({{ith_frame_id}}, {{jth_frame_id}}): {{len(good_matches)}} хороших совпадений, {{len(inliers)}} инлайеров.")
                else:
                    inlier_matches[(ith_frame_id, jth_frame_id)] = []
                    if verbose == 2:
                        print(f"Пара ({{ith_frame_id}}, {{jth_frame_id}}): недостаточно совпадений ({{len(good_matches)}})")
        
        with open(save_file, 'wb') as f:
            pickle.dump(inlier_matches, f)
        
    return inlier_matches


class UnionFind:
    def __init__(self):
        self.parent = {{}}
        
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


def buildTracks(inlier_matches, verbose=False):
    uf = UnionFind()
    
    for (frame_id_1, frame_id_2), matches in tqdm(inlier_matches.items(), desc="Составление графа"):
        for m in matches:
            obs1 = (frame_id_1, m[0])
            obs2 = (frame_id_2, m[1])
            uf.union(obs1, obs2)
    
    tracks = defaultdict(list)
    for obs in uf.parent.keys():
        root = uf.find(obs)
        tracks[root].append(obs)
    
    valid_tracks = {{}}
    bad_tracks = {{}}
    
    for track_id, obs_list in tqdm(tracks.items(), desc="Построение треков"):
        img_occurrences = defaultdict(int)
        conflict = False
        for obs in obs_list:
            frame_id, _ = obs
            img_occurrences[frame_id] += 1
            if img_occurrences[frame_id] > 1:
                
                conflict = True
                break
        if conflict or len(obs_list) < 2:
            bad_tracks[track_id] = obs_list
        else:
            valid_tracks[track_id] = sorted(obs_list, key=lambda pair: pair[0])

    if verbose:
        total_points = sum(len(v) for v in tracks.values())
        bad_points = sum(len(v) for v in bad_tracks.values())
        print(f'Отфильтровано {{bad_points}} наблюдений из {{total_points}} ({{bad_points/total_points:.2f}} от общего числа).')
        print(f'Всего треков: {{len(tracks)}}, хороших: {{len(valid_tracks)}}.')
    if verbose == 2:
        good_tracks_stats = Counter(len(v) for v in valid_tracks.values())
        sorted_good = sorted(good_tracks_stats.items(), key=lambda x: x[0])
        print("Распределение длин хороших треков (по возрастанию):", sorted_good)

        bad_tracks_stats = Counter(len(v) for v in bad_tracks.values())
        sorted_bad = sorted(bad_tracks_stats.items(), key=lambda x: x[0])
        print("Распределение длин плохих треков (по возрастанию):", sorted_bad)
    
    return list(valid_tracks.values())


def visualize_track(data_dir, valid_tracks, features_dir="./data", track_id=None, track_len=10, cols=5):
    rgb_path = Dataset.get_rgb_list_file(data_dir)
    assert os.path.isfile(rgb_path), "rgb.txt не найден в data_dir"
    path_list = Dataset.read_dict_of_lists(rgb_path, key_type=int)
    
    test_name = os.path.basename(os.path.normpath(data_dir))
    keypoints_by_img = {{}}
    for img_id in path_list:
        kp_file = os.path.join(features_dir, f'{{test_name}}_{{img_id}}_kp.pkl')
        if not os.path.isfile(kp_file):
            print(f"Файл с ключевыми точками для изображения {{img_id}} не найден!")
            continue
        with open(kp_file, 'rb') as f:
            kp_ser = pickle.load(f)
        keypoints_by_img[img_id] = deserialize_keypoints(kp_ser)
    
    if not valid_tracks:
        print("Нет треков для визуализации")
        return
    if track_id is None:
        track_iter = iter(valid_tracks.items())
        track_id, track_obs = next(track_iter)
        while len(valid_tracks[track_id]) < track_len:
            track_id, track_obs = next(track_iter)
    else:
        track_obs = valid_tracks.get(track_id, None)
        if track_obs is None:
            print(f"Трек {{track_id}} не найден.")
            return
    print(f"Визуализация трека {{track_id}} с наблюдениями: {{track_obs}}")
    
    n_obs = len(track_obs)
    rows = math.ceil(n_obs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.tolist()
    else:
        axes = [ax for row in axes for ax in row]
        
    for idx, (img_id, kp_idx) in tqdm(enumerate(track_obs), total=len(track_obs)):
        ax = axes[idx]
        if img_id not in path_list:
            print(f"Нет пути для изображения {{img_id}} в rgb.txt")
            continue
        img_path = os.path.join(data_dir, path_list[img_id])
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is None:
            print(f"Не удалось загрузить изображение: {{img_path}}")
            continue
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        if img_id not in keypoints_by_img:
            print(f"Ключевые точки для изображения {{img_id}} не загружены!")
            continue
        kp_list = keypoints_by_img[img_id]
        if kp_idx >= len(kp_list):
            print(f"Индекс ключевой точки {{kp_idx}} вне диапазона для изображения {{img_id}}")
            continue
        kp = kp_list[kp_idx]
        pt = (kp.pt[0], kp.pt[1])
        
        ax.imshow(img_rgb)
        ax.set_title(f"Изобр. {{img_id}} KP {{kp_idx}}")
        ax.axis("off")
        ax.scatter(pt[0], pt[1], s=30, c='red', marker='o')
    
    for i in range(n_obs, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()


def get_camera_center(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    center = - np.dot(R.T, t)
    return center


def triangulateTracks(valid_tracks, keypoints, known_poses, intrinsics, verbose=0):
    points_3d = []
    
    K = np.array([[intrinsics.fx, 0, intrinsics.cx],
                  [0, intrinsics.fy, intrinsics.cy],
                  [0, 0, 1]], dtype=np.float64)

    
    for track in tqdm(valid_tracks, desc="Триангуляция треков"):
        valid_frames = [frame_id for (frame_id, pt) in track if frame_id in known_poses]
        track = dict(track)

        if len(valid_frames) < 2:
            points_3d.append(None)
            if verbose == 2:
                print(f"Не удалось триангулировать трек -- недостаточно точек")
            continue
        
        max_distance = -1
        best_pair = None
    
        for frame_id_1, frame_id_2 in combinations(valid_frames, 2):
            T1 = known_poses[frame_id_1]
            T2 = known_poses[frame_id_2]
            center1 = get_camera_center(T1)
            center2 = get_camera_center(T2)
            distance = np.linalg.norm(center1 - center2)
            if distance > max_distance:
                max_distance = distance
                best_pair = (frame_id_1, frame_id_2)

        frame_id_1, frame_id_2 = best_pair
        T1 = known_poses[frame_id_1]
        T2 = known_poses[frame_id_2]
        pts1 = keypoints[frame_id_1][track[frame_id_1]].pt
        pts2 = keypoints[frame_id_2][track[frame_id_2]].pt

        tmp1 = -(T1[0:3, 0:3].T @ T1[0:3, 3].reshape(3, 1))
        tmp2 = -(T2[0:3, 0:3].T @ T2[0:3, 3].reshape(3, 1))
        P1 = K @ np.concatenate((T1[0:3, 0:3].T, tmp1), axis=1)
        P2 = K @ np.concatenate((T2[0:3, 0:3].T, tmp2), axis=1)

        point_4d = cv.triangulatePoints(P1, P2, np.array(pts1, ndmin=2).T, np.array(pts2, ndmin=2).T)
        point_3d = (point_4d / point_4d[3])[:3].flatten()
        
        points_3d.append(point_3d)

    return points_3d


def filterPointsByReprojectionError(points_3d, valid_tracks, keypoints, known_poses, intrinsics, params, verbose=0):
    filtered_tracks = []
    filtered_points_3d = []
    removed_count = 0

    K = np.array([[intrinsics.fx, 0, intrinsics.cx],
                  [0, intrinsics.fy, intrinsics.cy],
                  [0, 0, 1]], dtype=np.float64)

    for track_idx, track in tqdm(enumerate(valid_tracks), desc='Подсчёт ошибки репроекции'):
        point_3d = points_3d[track_idx]
        if point_3d is None:
            continue
        valid = True

        for (frame_id, kp_idx) in track:
            if frame_id not in known_poses:
                continue
            T = known_poses[frame_id]
            R = T[0:3, 0:3]
            t = T[0:3, 3]

            R_wc = R.T
            t_wc = -R_wc.dot(t)
            rvec, _ = cv.Rodrigues(R_wc)
            tvec = t_wc.reshape(3, 1)
            proj_point, _ = cv.projectPoints(point_3d.reshape(1, 3), rvec, tvec, K, None)
            proj_point = proj_point.reshape(-1)

            obs_pt = np.array(keypoints[frame_id][kp_idx].pt)
            error = np.linalg.norm(obs_pt - proj_point)
            if error > params.reproj_threshold:
                valid = False
                break
        if valid:
            filtered_tracks.append(track)
            filtered_points_3d.append(point_3d)
        else:
            removed_count += 1

    if verbose:
        total_tracks = len(valid_tracks)
        print(f"Отфильтровано {{removed_count}} треков из {{total_tracks}} по ошибке репроекции (>{{params.reproj_threshold}} px).")
        print(f"Осталось 3D-точек: {{len(filtered_points_3d)}}")
    return filtered_points_3d, filtered_tracks


def estimateUnknownCameraPoses(rgb_paths, descriptors, keypoints, known_poses, known_frames, filtered_points_3d, filtered_tracks, params, intrinsics, verbose=0):
    unknown_poses = {{}}
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    obs_to_point_idx = {{}}
    for point_idx, track in enumerate(filtered_tracks):
        for (frame_id, kp_idx) in track:
            obs_to_point_idx[(frame_id, kp_idx)] = point_idx

    unknown_frames = [fid for fid in rgb_paths.keys() if fid not in known_poses]
    count_localized = 0

    K = np.array([[intrinsics.fx, 0, intrinsics.cx],
                  [0, intrinsics.fy, intrinsics.cy],
                  [0, 0, 1]], dtype=np.float64)

    for frame_id in tqdm(unknown_frames, desc='Вычисление позиций камер'):
        des_unknown = descriptors[frame_id]
        kp_unknown = keypoints[frame_id]
        if des_unknown is None:
            continue

        object_points = []
        image_points = []
        used_point_indices = set()

        for ref_id in known_frames:
            des_ref = descriptors[ref_id]
            kp_ref = keypoints[ref_id]
            if des_ref is None:
                continue
            matches = bf.knnMatch(des_unknown, des_ref, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < params.lowe_ratio_threshold * n.distance:
                    good_matches.append(m)
            for m in good_matches:
                obs = (ref_id, m.trainIdx)
                if obs in obs_to_point_idx:
                    pt_idx = obs_to_point_idx[obs]
                    if pt_idx in used_point_indices:
                        continue
                    used_point_indices.add(pt_idx)
                    object_points.append(filtered_points_3d[pt_idx])
                    
                    x, y = kp_unknown[m.queryIdx].pt
                    image_points.append([x, y])
        if len(object_points) < 4:
            if verbose:
                print(f"Кадр {{frame_id}}: недостаточно соответствий для PnP ({{len(object_points)}} точек).")
            continue

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        success, rvec, tvec, inliers = cv.solvePnPRansac(object_points, image_points, K, None)
        if not success:
            if verbose:
                print(f"Кадр {{frame_id}}: solvePnPRansac не нашёл решение.")
            continue

        R_wc, _ = cv.Rodrigues(rvec)
        t_wc = tvec.reshape(3,)
        R_cw = R_wc.T
        t_cw = -R_wc.T.dot(t_wc)
        pose_matrix = np.eye(4)
        pose_matrix[0:3, 0:3] = R_cw
        pose_matrix[0:3, 3] = t_cw

        unknown_poses[frame_id] = pose_matrix
        count_localized += 1
        if verbose:
            inlier_count = len(inliers) if inliers is not None else 0
            print(f"Кадр {{frame_id}}: локализован по {{inlier_count}} инлайерам из {{len(object_points)}} соответствий.")

    if verbose:
        total_unknown = len(unknown_frames)
        print(f"Локализовано {{count_localized}} из {{total_unknown}} неизвестных камер.")
    return unknown_poses

def estimate_trajectory(data_dir, out_dir, features_dir='./data', reCompute=True, params=None, verbose=0):
    if params is None:
        params = SimpleNamespace(
            ORBCount={opt_params.ORBCount},
            ransac_reproj_threshold=2.7
            confidence_level=0.99,
            lowe_ratio_threshold={opt_params.lowe_ratio_threshold},
            reproj_threshold=10.0
        )
    # 1. Загрузка данных
    rgb_paths = Dataset.read_dict_of_lists(Dataset.get_rgb_list_file(data_dir))
    known_poses = Trajectory.read(Dataset.get_known_poses_file(data_dir))

    intrinsics_file = Dataset.get_intrinsics_file(data_dir)
    if not os.path.isfile(intrinsics_file):
        print("Файл intrinsics.txt не найден в data_dir")
    intrinsics = Intrinsics.read(intrinsics_file)

    # 2. Вычисление ORB дескрипторов
    keypoints, descriptors = computeORB(data_dir=data_dir, features_dir=features_dir, params=params, reCompute=reCompute)

    # 3. Вычисление инлаеров
    known_frames = list(known_poses.keys())
    inlier_matches = computeInliers(keypoints, descriptors, known_frames, params, data_dir, features_dir, verbose=verbose, reCompute=reCompute)

    # 4. Вычисление треков
    valid_tracks = buildTracks(inlier_matches, verbose=verbose)

    # визуализация треков, если хотим
    #for key in valid_tracks.keys():
        #visualize_track(data_dir=data_dir, valid_tracks=valid_tracks, track_id=key, cols=4)

    # 5. Триангуляция точек
    points_3d = triangulateTracks(valid_tracks, keypoints, known_poses, intrinsics, verbose=verbose)

    # 6. Фильтрация 3D-точек по ошибке репроекции
    filtered_points_3d, filtered_tracks = filterPointsByReprojectionError(points_3d, valid_tracks, keypoints, known_poses, intrinsics, params, verbose=verbose)

    # 7. Вычисление позиций неизвестных камер
    unknown_poses = estimateUnknownCameraPoses(rgb_paths, descriptors, keypoints, known_poses, known_frames, filtered_points_3d, filtered_tracks, params, intrinsics, verbose=verbose)

    Trajectory.write(Dataset.get_result_poses_file(out_dir), unknown_poses)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: {{}} <data_dir> <out_dir>".format(sys.argv[0]))
        exit(1)
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    estimate_trajectory(data_dir, out_dir)
"""
    with open(filename, 'w') as f:
        f.write(code)

def optimize_hyperparameters(n_trials=100, timeout=3600 * 6):
    """
    Запускает оптимизацию гиперпараметров с использованием Optuna.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print("\nЛучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"Лучшая оценка: {study.best_value}")
    
    df = study.trials_dataframe()
    df.to_csv('optimization_history.csv', index=False)
    
    return study.best_params, study


if __name__ == "__main__":
    start_time = time.time()
    best_params, best_score = optimize_hyperparameters(n_trials=100, timeout=3600*5)  # максимум 5 часов
    знend_time = time.time()
    
    print(f"\nОптимизация заняла {(end_time - start_time) / 60:.2f} минут")
    print(f"Лучшая оценка: {best_score}")
    print("Лучшие параметры:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")