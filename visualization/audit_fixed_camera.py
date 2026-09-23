#!/usr/bin/env python3
"""Audit sampled background image motion, without treating it as calibration.

The background ROI is explicit and must be visually verified to exclude people.
For these three recordings the upper 26% is ceiling, verified on contact sheets.
Lucas-Kanade tracks use forward/backward filtering and RANSAC; report pixel-space
displacements, not a recovered camera pose or a claim of exact camera immobility.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def measure(a, b, roi_fraction, scale):
    mask = np.zeros_like(a)
    # The right foreground can contain a close-up head in recordings 1 and 2.
    mask[8:int(a.shape[0] * roi_fraction), 8:int(a.shape[1] * .82)] = 255
    pa = cv2.goodFeaturesToTrack(a, 600, .002, 5, mask=mask, blockSize=5)
    if pa is None or len(pa) < 12:
        return {'status': 'insufficient_features'}, None
    pb, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, pa, None, winSize=(25, 25), maxLevel=3,
                                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .001))
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(b, a, pb, None, winSize=(25, 25), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .001))
    good = ok.ravel().astype(bool) & ok2.ravel().astype(bool) & (np.linalg.norm(back - pa, axis=(1, 2)) < .6)
    x, y = pa[good].reshape(-1, 2), pb[good].reshape(-1, 2)
    if len(x) < 12:
        return {'status': 'insufficient_bidirectional_tracks', 'tracks': len(x)}, None
    matrix, inliers = cv2.estimateAffinePartial2D(x, y, method=cv2.RANSAC,
                                                 ransacReprojThreshold=.75, maxIters=4000, confidence=.999)
    if matrix is None:
        return {'status': 'ransac_failed'}, None
    inliers = inliers.ravel().astype(bool)
    displacement = np.linalg.norm(y[inliers] - x[inliers], axis=1) / scale
    residual = np.linalg.norm((x @ matrix[:, :2].T + matrix[:, 2] - y)[inliers], axis=1) / scale
    center = np.array([(8 + a.shape[1] * .82) / 2, (8 + a.shape[0] * roi_fraction) / 2])
    center_flow = (matrix[:, :2] @ center + matrix[:, 2] - center) / scale
    angle = np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))
    tiles = np.floor(x[inliers] / np.array([a.shape[1] / 4, a.shape[0] * roi_fraction / 2])).astype(int)
    report = dict(status='ok', tracked_candidates=len(x), ransac_inliers=int(inliers.sum()),
                  inlier_fraction=float(inliers.mean()), roi_tiles_covered=int(len(np.unique(tiles, axis=0))),
                  median_inlier_displacement_original_px=float(np.median(displacement)),
                  p95_inlier_displacement_original_px=float(np.percentile(displacement, 95)),
                  affine_roi_center_displacement_xy_original_px=center_flow.tolist(),
                  affine_roi_center_displacement_original_px=float(np.linalg.norm(center_flow)),
                  reference_point_original_px=(center / scale).tolist(),
                  affine_rotation_deg=float(angle), affine_scale=float(np.linalg.norm(matrix[:, 0])),
                  median_fit_residual_original_px=float(np.median(residual)))
    return report, (x, y, inliers)


def audit(path, output, roi_fraction=.26, sample_s=1.):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1., 812 / sw)
    indices = np.unique(np.r_[np.arange(0, count, max(1, round(sample_s * fps))), count - 1]).astype(int)
    grays, colors, records, pictures = [], [], [], []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if not ok:
            raise ValueError(f'Failed frame {index} of {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (round(sw * scale), round(sh * scale)), interpolation=cv2.INTER_AREA)
        colors.append(rgb)
        grays.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    cap.release()
    for i in range(1, len(indices)):
        for kind, j in [('adjacent_sample', i - 1), ('first_frame_anchor', 0)]:
            result, tracks = measure(grays[j], grays[i], roi_fraction, scale)
            result.update(pair_kind=kind, frame_a=int(indices[j]), frame_b=int(indices[i]), timestamp_b_s=float(indices[i] / fps))
            records.append(result)
            if kind == 'first_frame_anchor' and i in {1, len(indices) // 2, len(indices) - 1}:
                im = Image.fromarray(colors[i])
                d = ImageDraw.Draw(im)
                d.rectangle((0, 0, int(im.width * .82), int(im.height * roi_fraction)), outline=(43, 208, 221), width=3)
                if tracks:
                    a, b, keep = tracks
                    for p, q, ok in zip(a, b, keep):
                        color = (43, 220, 165) if ok else (255, 109, 99)
                        d.ellipse((q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2), fill=color)
                        d.line([tuple(q), tuple(q + (q - p) * 10)], fill=color, width=1)
                d.rectangle((0, im.height - 65, im.width, im.height), fill=(13, 21, 31))
                d.text((12, im.height - 55), f'Ceiling ROI: upper {100 * roi_fraction:.0f}%, left 82% | frame {indices[i]} | flow arrows x10', fill='white')
                if result['status'] == 'ok':
                    d.text((12, im.height - 31), f'Anchor displacement median {result["median_inlier_displacement_original_px"]:.3f} px | inliers {result["ransac_inliers"]}', fill=(43, 208, 221))
                pictures.append(im)
    good = [r for r in records if r['status'] == 'ok' and r['ransac_inliers'] >= 20 and r['inlier_fraction'] >= .65 and r['roi_tiles_covered'] >= 3]
    def summarise(kind):
        selected = [r for r in good if r['pair_kind'] == kind]
        if not selected:
            return None
        values = np.array([r['affine_roi_center_displacement_original_px'] for r in selected])
        return dict(pairs=len(selected), median_center_displacement_original_px=float(np.median(values)),
                    p95_center_displacement_original_px=float(np.percentile(values, 95)),
                    max_center_displacement_original_px=float(values.max()),
                    max_abs_rotation_deg=float(max(abs(r['affine_rotation_deg']) for r in selected)),
                    median_inlier_fraction=float(np.median([r['inlier_fraction'] for r in selected])))
    adjacent, anchor = summarise('adjacent_sample'), summarise('first_frame_anchor')
    support_fraction = len(good) / max(1, len(records))
    verdict = 'inconclusive'
    if support_fraction >= .85 and anchor:
        verdict = 'consistent_with_fixed_camera_at_sampled_times' if anchor['p95_center_displacement_original_px'] <= 1. and anchor['max_center_displacement_original_px'] <= 2. else 'background_motion_detected_review_required'
    report = dict(video=str(path.resolve()), source_dimensions=[sw, sh], fps=fps, source_frame_count=count,
                  sample_interval_s=sample_s, sampled_frames=indices.tolist(), analysis_resize_scale=scale,
                  background_roi=dict(type='explicit_ceiling_region', normalized_y_range=[0, roi_fraction], normalized_x_range=[0, .82],
                                      selection_basis='Visually checked contact sheets; right foreground omitted to exclude close-up heads'),
                  method='Shi-Tomasi / bidirectional Lucas-Kanade / partial affine RANSAC',
                  verdict=verdict, supported_pair_fraction=support_fraction,
                  adjacent_sample_summary=adjacent, first_frame_anchor_summary=anchor,
                  limitations=['Sampled image-space background consistency only; not an extrinsic calibration',
                               'Unsampled short camera motion and subpixel motion cannot be ruled out',
                               'Each recording is audited separately; no shared extrinsic across videos is assumed'], pairs=records)
    output.mkdir(parents=True, exist_ok=True)
    target = output / (path.stem + '.camera_staticity.json')
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    sheet = Image.new('RGB', (pictures[0].width, pictures[0].height * len(pictures)))
    for i, picture in enumerate(pictures):
        sheet.paste(picture, (0, i * picture.height))
    sheet.save(output / (path.stem + '.background_tracks.jpg'), quality=92)
    print(json.dumps({'video': path.name, 'verdict': verdict, 'anchor': anchor}), flush=True)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('videos', type=Path, nargs='+')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ceiling-fraction', type=float, default=.26)
    ap.add_argument('--sample-s', type=float, default=1.)
    args = ap.parse_args()
    reports = [audit(path, args.output, args.ceiling_fraction, args.sample_s) for path in args.videos]
    (args.output / 'camera_staticity_summary.json').write_text(json.dumps([
        {k: r[k] for k in ('video', 'verdict', 'supported_pair_fraction', 'adjacent_sample_summary', 'first_frame_anchor_summary', 'limitations')}
        for r in reports], indent=2) + '\n', encoding='utf-8')
