from common.dataset import Dataset
from common.trajectory import Trajectory
from common.intrinsics import Intrinsics

import os
import glob
import math
import pickle
import cv2 as cv
import numpy as np
#from tqdm import tqdm
#from types import SimpleNamespace
from itertools import combinations
#from matplotlib import pyplot as plt
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



def computeORB(data_dir, features_dir, params, reCompute=True):
    rgb_path = Dataset.get_rgb_list_file(data_dir)
    assert os.path.isfile(rgb_path), "rgb.txt не найден в data_dir"
    test_name = data_dir.split('/')[-1]
    print(test_name)

    keypoints = {}
    descriptors = {}

    path_list = Dataset.read_dict_of_lists(rgb_path, key_type=int)
    for frame_id, path in path_list.items(): #tqdm(path_list.items(), desc="Вычисление дескрипторов"):
        des_path = os.path.join(features_dir, f'{test_name}_{frame_id}_des.pkl')
        kp_path = os.path.join(features_dir, f'{test_name}_{frame_id}_kp.pkl')
        img_path = os.path.join(data_dir, path)

        if not reCompute and os.path.exists(des_path) and os.path.exists(kp_path):
            pass
            #with open(des_path, 'rb') as f:
            #    descriptors[frame_id] = pickle.load(f)
            #with open(kp_path, 'rb') as f:
            #    keypoints_serialised = pickle.load(f)
            #    keypoints[frame_id] = deserialize_keypoints(keypoints_serialised)
        else:
            img_bgr = cv.imread(img_path, cv.IMREAD_COLOR)
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

            orb = cv.ORB_create(nfeatures=params['ORBCount'])
            kp = orb.detect(img_rgb, None)
            kp, des = orb.compute(img_rgb, kp)

            descriptors[frame_id] = des
            keypoints[frame_id] = kp

            #img_upd = cv.drawKeypoints(img_rgb, kp, None, color=(0,255,0), flags=0)
            
            #with open(des_path, 'wb') as f:
            #    pickle.dump(des, f)

            #keypoints_serialised = serialize_keypoints(kp)
            #with open(kp_path, 'wb') as f:
            #    pickle.dump(keypoints_serialised, f)
    return keypoints, descriptors


def computeInliers(keypoints, descriptors, known_frames, params, data_dir, features_dir='./data', verbose=0, reCompute=True):
    test_name = data_dir.split('/')[-1]
    save_file = os.path.join(features_dir, f'{test_name}_inlier_matches.pkl')

    if not reCompute and os.path.exists(save_file):
        pass
        #with open(save_file, 'rb') as f:
        #    inlier_matches = pickle.load(f)
    else:
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        
        inlier_matches = {}
        
        for i in range(len(known_frames)): # tqdm(range(len(known_frames)), desc="Матчинг инлайеров"):
            ith_frame_id = known_frames[i]
            des_i = descriptors[ith_frame_id]
            kp_i = keypoints[ith_frame_id]
            
            for j in range(i + 1, len(known_frames)):
                jth_frame_id = known_frames[j]
                des_j = descriptors[jth_frame_id]
                kp_j = keypoints[jth_frame_id]
                
                matches = bf.knnMatch(des_i, des_j, k=2)

                ratio_thresh = params['lowe_ratio_threshold']
                good_matches = []
                for m, n in matches:
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) >= 8:
                    pts_i = np.float32([kp_i[m.queryIdx].pt for m in good_matches])
                    pts_j = np.float32([kp_j[m.trainIdx].pt for m in good_matches])

                    F, mask = cv.findFundamentalMat(pts_i, pts_j, cv.FM_RANSAC, params['ransac_reproj_threshold'], params['confidence_level'])
                    if mask is not None:
                        inliers = []
                        for k, match in enumerate(good_matches):
                            if mask[k][0]:
                                inliers.append((match.queryIdx, match.trainIdx))

                    inlier_matches[(ith_frame_id, jth_frame_id)] = inliers
                    if verbose == 2:
                        print(f"Пара ({ith_frame_id}, {jth_frame_id}): {len(good_matches)} хороших совпадений, {len(inliers)} инлайеров.")
                else:
                    inlier_matches[(ith_frame_id, jth_frame_id)] = []
                    if verbose == 2:
                        print(f"Пара ({ith_frame_id}, {jth_frame_id}): недостаточно совпадений ({len(good_matches)})")
        
        #with open(save_file, 'wb') as f:
        #    pickle.dump(inlier_matches, f)
        
    return inlier_matches


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


def buildTracks(inlier_matches, verbose=False):
    uf = UnionFind()
    
    for (frame_id_1, frame_id_2), matches in inlier_matches.items(): #tqdm(inlier_matches.items(), desc="Составление графа"):
        for m in matches:
            obs1 = (frame_id_1, m[0])
            obs2 = (frame_id_2, m[1])
            uf.union(obs1, obs2)
    
    tracks = defaultdict(list)
    for obs in uf.parent.keys():
        root = uf.find(obs)
        tracks[root].append(obs)
    
    valid_tracks = {}
    bad_tracks = {}
    
    for track_id, obs_list in tracks.items(): # tqdm(tracks.items(), desc="Построение треков"):
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
        print(f'Отфильтровано {bad_points} наблюдений из {total_points} ({bad_points/total_points:.2f} от общего числа).')
        print(f'Всего треков: {len(tracks)}, хороших: {len(valid_tracks)}.')
    if verbose == 2:
        good_tracks_stats = Counter(len(v) for v in valid_tracks.values())
        sorted_good = sorted(good_tracks_stats.items(), key=lambda x: x[0])
        print("Распределение длин хороших треков (по возрастанию):", sorted_good)

        bad_tracks_stats = Counter(len(v) for v in bad_tracks.values())
        sorted_bad = sorted(bad_tracks_stats.items(), key=lambda x: x[0])
        print("Распределение длин плохих треков (по возрастанию):", sorted_bad)
    
    return list(valid_tracks.values())


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

    
    for track in valid_tracks: # tqdm(valid_tracks, desc="Триангуляция треков"):
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

    for track_idx, track in enumerate(valid_tracks): # tqdm(enumerate(valid_tracks), desc='Подсчёт ошибки репроекции'):
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
            if error > params['reproj_threshold']:
                valid = False
                break
        if valid:
            filtered_tracks.append(track)
            filtered_points_3d.append(point_3d)
        else:
            removed_count += 1

    if verbose:
        total_tracks = len(valid_tracks)
        print(f"Отфильтровано {removed_count} треков из {total_tracks} по ошибке репроекции (>{params['reproj_threshold']} px).")
        print(f"Осталось 3D-точек: {len(filtered_points_3d)}")
    return filtered_points_3d, filtered_tracks


def estimateUnknownCameraPoses(rgb_paths, descriptors, keypoints, known_poses, known_frames, filtered_points_3d, filtered_tracks, params, intrinsics, verbose=0):
    unknown_poses = {}
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    obs_to_point_idx = {}
    for point_idx, track in enumerate(filtered_tracks):
        for (frame_id, kp_idx) in track:
            obs_to_point_idx[(frame_id, kp_idx)] = point_idx

    unknown_frames = [fid for fid in rgb_paths.keys() if fid not in known_poses]
    count_localized = 0

    K = np.array([[intrinsics.fx, 0, intrinsics.cx],
                  [0, intrinsics.fy, intrinsics.cy],
                  [0, 0, 1]], dtype=np.float64)

    for frame_id in unknown_frames: # tqdm(unknown_frames, desc='Вычисление позиций камер'):
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
                if m.distance < params['lowe_ratio_threshold'] * n.distance:
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
                print(f"Кадр {frame_id}: недостаточно соответствий для PnP ({len(object_points)} точек).")
            continue

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        success, rvec, tvec, inliers = cv.solvePnPRansac(object_points, image_points, K, None)
        if not success:
            if verbose:
                print(f"Кадр {frame_id}: solvePnPRansac не нашёл решение.")
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
            print(f"Кадр {frame_id}: локализован по {inlier_count} инлайерам из {len(object_points)} соответствий.")

    if verbose:
        total_unknown = len(unknown_frames)
        print(f"Локализовано {count_localized} из {total_unknown} неизвестных камер.")
    return unknown_poses

def estimate_trajectory(data_dir, out_dir, features_dir='./data', reCompute=True, params=None, verbose=0):
    if params is None:
        params = {
            'ORBCount' : 1000,                 # количество вычисляемых ORB дескрипторов
            'ransac_reproj_threshold' : 3.0,   # допустимая ошибка репроекции для RANSAC
            'confidence_level' : 0.99,         # уровень уверенности для RANSAC
            'lowe_ratio_threshold' : 0.51,     # порог для теста Лоу
            'reproj_threshold' : 10.0,         # допустимая ошибка репроекции для опорных точек
        }
        '''params = SimpleNamespace(
            ORBCount=1000,                 # количество вычисляемых ORB дескрипторов
            ransac_reproj_threshold=3.0,   # допустимая ошибка репроекции для RANSAC
            confidence_level=0.99,         # уровень уверенности для RANSAC
            lowe_ratio_threshold=0.51,      # порог для теста Лоу
            reproj_threshold=10.0,          # допустимая ошибка репроекции для опорных точек
        )'''
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
    estimate_trajectory(data_dir= './tests/00_test_slam_input', features_dir='./data', out_dir='./results', reCompute=True, verbose=0)

