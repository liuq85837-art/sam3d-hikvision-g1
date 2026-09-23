"""Audit a COMPLETE real SAM3D sequence; write diagnostics only, never alter motion."""
from pathlib import Path
import argparse
import json
import hashlib

import cv2
import numpy as np
from scipy.signal import savgol_filter

from fixed_camera_world import closure_metrics


def stats(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)),
                p05=float(np.percentile(a, 5)), p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)), max=float(a.max()), std=float(a.std()))


def derivatives(p, t, valid):
    dt = np.diff(t)
    ok = valid[:-1] & valid[1:] & (dt < max(.1, 2.5*np.median(dt)))
    v = np.diff(p, axis=0) / dt[:, None]
    a = np.diff(v, axis=0) / ((dt[:-1]+dt[1:])/2)[:, None]
    acc_ok = ok[:-1] & ok[1:]
    return dict(speed_m_s=stats(np.linalg.norm(v[ok], axis=1)),
                acceleration_m_s2=stats(np.linalg.norm(a[acc_ok], axis=1)),
                vertical_speed_m_s=stats(abs(v[ok, 2])),
                contiguous_path_m=float(np.linalg.norm(np.diff(p, axis=0)[ok], axis=1).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence', type=Path)
    parser.add_argument('--expected-frames', type=int, required=True)
    args = parser.parse_args()
    seq = args.sequence
    status = json.loads((seq/'camera/status.json').read_text(encoding='utf-8'))
    if status.get('status') != 'complete' or status.get('selected') != args.expected_frames:
        raise ValueError('Final inference is not complete for the expected selection; do not audit smoke aggregates')
    with np.load(seq/'camera/camera_sequence.npz', allow_pickle=False) as z:
        c = {k: z[k] for k in z.files}
    with np.load(seq/'world/world_joints.npz', allow_pickle=False) as z:
        w = {k: z[k] for k in z.files}
    if len(c['joints_camera']) != args.expected_frames or len(w['joints_world']) != args.expected_frames:
        raise ValueError('Aggregate count does not match final status')
    gauge = json.loads((seq/'world/gauge.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256((seq/'camera/camera_sequence.npz').read_bytes()).hexdigest()
    if digest != gauge['source_sha256']:
        raise ValueError('World gauge does not bind the final camera aggregate')
    out = seq/'quality'
    out.mkdir(exist_ok=True)
    t = c['timestamps_s'].astype(float)
    frames = c['frame_indices']
    valid = c['valid'] & w['valid']
    camera = c['joints_camera'].astype(float)
    world = w['joints_world'].astype(float)
    pelvis = world[:, [9,10]].mean(1)
    camera_pelvis = camera[:, [9,10]].mean(1)
    uv = c['keypoints_2d']
    bones = {'left_upper_arm': (5,7), 'right_upper_arm': (6,8), 'left_forearm': (7,62), 'right_forearm': (8,41),
             'left_thigh': (9,11), 'right_thigh': (10,12), 'left_shin': (11,13), 'right_shin': (12,14),
             'hip_width': (9,10), 'shoulder_width': (5,6)}
    bone_report = {}
    for name, (a,b) in bones.items():
        length = np.linalg.norm(world[:, a]-world[:, b], axis=1)[valid]
        median = np.median(length)
        bone_report[name] = {**stats(length), 'coefficient_of_variation': float(length.std()/length.mean()),
                             'robust_sigma_over_median': float(1.4826*np.median(abs(length-median))/median)}
    foot = world[:, [15,16,17,18,19,20], 2]
    low = foot.min(1)
    raw_derivative = derivatives(pelvis, t, valid)
    step = np.linalg.norm(np.diff(pelvis, axis=0), axis=1)
    contiguous=valid[:-1]&valid[1:]&np.isfinite(step)&(np.diff(t)<max(.1,2.5*np.median(np.diff(t))))
    step_ids=np.flatnonzero(contiguous)
    spikes=step_ids[np.argsort(step[step_ids])[-10:][::-1]]
    report = {
        'status': 'complete_real_sequence_audited', 'source_camera_sha256': digest,
        'selected_frames': len(t), 'valid_frames': int(valid.sum()), 'world_frame_id': gauge['world_frame_id'],
        'source_frames_first_last': [int(frames[0]), int(frames[-1])], 'source_times_first_last_s': [float(t[0]),float(t[-1])],
        'median_sample_interval_s': float(np.median(np.diff(t))), 'scale_status': gauge['scale_status'],
        'camera_pelvis_depth_model_m': stats(camera_pelvis[valid,2]),
        'world_pelvis_height_model_m': stats(pelvis[valid,2]),
        'root_motion': raw_derivative,
        'lowest_predicted_foot_height_model_m': stats(low[valid]),
        'lowest_foot_below_minus_0p03_fraction': float(np.mean(low[valid] < -.03)),
        'lowest_foot_above_0p03_fraction': float(np.mean(low[valid] > .03)),
        'bone_lengths_model_m': bone_report,
        'inferred_ground': gauge['diagnostics'],
        'largest_root_steps': [{'from_frame': int(frames[i]),'to_frame': int(frames[i+1]),
                                'time_s': float(t[i]), 'step_model_m': float(step[i]),
                                'speed_model_m_s': float(step[i]/(t[i+1]-t[i]))} for i in spikes],
        'loop_constraint_applied': False, 'original_data_modified': False,
        'foot_contact_truth_available': False,
    }
    ns = np.rint(t*1e9).astype(np.int64)
    report['raw_closure'] = closure_metrics(pelvis, ns, valid, frame_indices=frames)
    valid_indices=np.flatnonzero(valid)
    start = valid & (t <= t[valid_indices[0]]+.25)
    end = valid & (t >= t[valid_indices[-1]]-.25)
    report['endpoint_projected_hip_pixels'] = {
        'start': np.median(uv[start][:,[9,10]].mean(1),axis=0).tolist(),
        'end': np.median(uv[end][:,[9,10]].mean(1),axis=0).tolist()}
    report['endpoint_projected_foot_pixels'] = {
        'start': np.median(uv[start][:,[15,16,17,18,19,20]].mean(1),axis=0).tolist(),
        'end': np.median(uv[end][:,[15,16,17,18,19,20]].mean(1),axis=0).tolist()}
    if valid.all():
        smooth = savgol_filter(pelvis, 5, 2, axis=0, mode='interp')
        delta = np.linalg.norm(smooth-pelvis,axis=1)
        report['diagnostic_smoothing_only_not_exported'] = {
            'method': 'offline Savitzky-Golay root only; window 5 samples, polynomial 2; fixed original gauge',
            'motion': derivatives(smooth,t,valid), 'root_shift_model_m': stats(delta),
            'closure': closure_metrics(smooth,ns,valid,frame_indices=frames),
            'note': 'Only a diagnostic comparison. No motion output overwritten; no endpoint or contact constraints.'}
    (out/'world_reconstruction_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    metadata = json.loads(str(c['metadata']))
    cap = cv2.VideoCapture(metadata['source_video'])
    first_frame=int(frames[np.flatnonzero(valid)[0]])
    last_frame=int(frames[np.flatnonzero(valid)[-1]])
    third_second=max(1,round(metadata['source_fps']/3))
    selected=[first_frame,min(first_frame+third_second,last_frame),
              min(first_frame+2*third_second,last_frame),
              int(frames[np.flatnonzero(valid)[int(valid.sum()*.435)]]),
              max(first_frame,last_frame-4*third_second),max(first_frame,last_frame-2*third_second),
              max(first_frame,last_frame-third_second),last_frame]
    panels = []
    for frame in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,im = cap.read()
        if not ok:
            continue
        im = cv2.resize(im,(650,496))
        cv2.putText(im,f'frame {frame} / {frame/metadata["source_fps"]:.2f} s',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,240),2,cv2.LINE_AA)
        panels.append(im)
    if len(panels)==8:
        grid=np.vstack([np.hstack(panels[:4]),np.hstack(panels[4:])])
        ok, encoded = cv2.imencode('.jpg', grid)
        if not ok:
            raise IOError('Could not encode endpoint evidence image')
        (out/'endpoint_video_evidence.jpg').write_bytes(encoded.tobytes())
    else:
        raise IOError(f'Could only decode {len(panels)}/8 endpoint evidence frames')
    cap.release()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes[0,0].plot(pelvis[:,0],pelvis[:,1],color='#169ca5',linewidth=1)
    axes[0,0].scatter([pelvis[0,0],pelvis[-1,0]],[pelvis[0,1],pelvis[-1,1]],c=['green','red'],s=30)
    axes[0,0].set(title='Raw pelvis XY; model metres',xlabel='world x',ylabel='world y',aspect='equal')
    for k,label in enumerate(['x','y','z']):
        axes[0,1].plot(t,pelvis[:,k],label=label,linewidth=1)
    axes[0,1].legend();axes[0,1].set(title='Pelvis coordinates',xlabel='source time (s)',ylabel='model metres')
    axes[1,0].plot(t,low,label='lowest predicted foot',linewidth=1)
    axes[1,0].axhline(0,color='black',linewidth=.8)
    axes[1,0].set(title='Inferred floor residual; not contact truth',xlabel='source time (s)',ylabel='model metres')
    for name,(a,b) in list(bones.items())[4:8]:
        axes[1,1].plot(t,np.linalg.norm(world[:,a]-world[:,b],axis=1),label=name,linewidth=.8)
    axes[1,1].legend(fontsize=8);axes[1,1].set(title='Leg segment lengths',xlabel='source time (s)',ylabel='model metres')
    fig.savefig(out/'world_reconstruction_audit.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'audit_json':str(out/'world_reconstruction_audit.json'), 'root_motion':raw_derivative,
                      'foot_height':report['lowest_predicted_foot_height_model_m'],
                      'depth':report['camera_pelvis_depth_model_m']},indent=2))


if __name__ == '__main__':
    main()
