"""Bounded offline root-only temporal candidate; no contact or loop constraints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt

from fixed_camera_world import closure_metrics, json_write


def summary(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    if not len(a):return None
    return {'median':float(np.median(a)), 'p95':float(np.percentile(a,95)), 'max':float(a.max())}


def segments(t,valid):
    dt=np.diff(t);max_gap=max(.1,float(np.median(dt))*2.5)
    starts=np.flatnonzero(valid & np.r_[True,~valid[:-1] | (dt>max_gap)])
    ends=np.flatnonzero(valid & np.r_[~valid[1:] | (dt>max_gap),True])+1
    return list(zip(starts,ends)),max_gap


def smooth_root(p,t,valid,method):
    result=p.copy(); blocks,max_gap=segments(t,valid);notes=[]
    for start,stop in blocks:
        part=p[start:stop]; times=t[start:stop];n=len(part)
        window=5 if method=='savgol5_poly3' else 7
        minimum=window if method.startswith('savgol') else 10
        if n<minimum:
            notes.append({'start':int(start),'stop_exclusive':int(stop),'status':'short_segment_unchanged'})
            continue
        grid=np.linspace(times[0],times[-1],n)
        irregular=float(np.max(abs(np.diff(times)/np.median(np.diff(times))-1)))>.02
        work=np.column_stack([np.interp(grid,times,part[:,d]) for d in range(3)]) if irregular else part
        if method.startswith('savgol'):
            filtered=savgol_filter(work,window,3,axis=0,mode='interp')
        elif method=='butterworth3hz_order2':
            sample_rate=1/float(np.mean(np.diff(grid)))
            if 3>=sample_rate/2:
                notes.append({'start':int(start),'stop_exclusive':int(stop),'status':'sample_rate_too_low_unchanged'})
                continue
            sos=butter(2,3,fs=sample_rate,output='sos')
            filtered=sosfiltfilt(sos,work,axis=0)
        else:raise ValueError(method)
        result[start:stop]=np.column_stack([np.interp(times,grid,filtered[:,d]) for d in range(3)]) if irregular else filtered
        notes.append({'start':int(start),'stop_exclusive':int(stop),'status':'filtered',
                      'temporary_uniform_resampling':irregular})
    return result,notes,max_gap


def metrics(joints,t,valid):
    p=joints[:,[9,10]].mean(1);dt=np.diff(t)
    blocks,max_gap=segments(t,valid)
    pair=valid[:-1]&valid[1:]&(dt<=max_gap)
    v=np.diff(p,axis=0)/dt[:,None]
    a=np.diff(v,axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
    foot=joints[:,[15,16,17,18,19,20],2].min(1)[valid]
    ns=np.rint((t-t[0])*1e9).astype(np.int64)
    return {'root_speed_model_m_s':summary(np.linalg.norm(v[pair],axis=1)),
            'root_acceleration_model_m_s2':summary(np.linalg.norm(a[pair[:-1]&pair[1:]],axis=1)),
            'path_length_model_m':float(np.linalg.norm(np.diff(p,axis=0)[pair],axis=1).sum()),
            'lowest_foot_abs_height_model_m':summary(abs(foot)),
            'lowest_foot_height_min_model_m':float(foot.min()),
            'lowest_foot_below_minus_3cm_fraction':float(np.mean(foot<-.03)),
            'closure':closure_metrics(p,ns,valid)}


def refine(input_path,output,max_rms=.05):
    input_path=Path(input_path);output=Path(output)
    if input_path.resolve()==(output/'world_joints.npz').resolve():
        raise ValueError('Temporal output must not overwrite its source')
    with np.load(input_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    raw=np.asarray(data['joints_world'],float)
    if raw.shape[1:]!=(70,3):raise ValueError('Expected MHR70 [T,70,3]')
    valid=np.asarray(data['valid'],bool)&np.isfinite(raw).all(axis=(1,2))
    t=np.asarray(data['timestamps_s'],float)
    if len(t)!=len(raw) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Invalid source timing')
    if valid.sum()<10:raise ValueError('Too few valid frames for a temporal candidate')
    root=raw[:,[9,10]].mean(1);before=metrics(raw,t,valid)
    reports=[];accepted=[]
    for method in ['savgol5_poly3','savgol7_poly3','butterworth3hz_order2']:
        smoothed,blocks,max_gap=smooth_root(root,t,valid,method)
        delta=smoothed-root;delta[~valid]=0.
        candidate=raw+delta[:,None]
        after=metrics(candidate,t,valid)
        magnitudes=np.linalg.norm(delta[valid],axis=1)
        rms=float(np.sqrt(np.mean(magnitudes**2)))
        first=int(np.flatnonzero(valid)[0]);last=int(np.flatnonzero(valid)[-1])
        endpoint_vector_change=np.asarray(after['closure']['endpoint_displacement_xyz_m'])-np.asarray(before['closure']['endpoint_displacement_xyz_m'])
        accel_ratio=after['root_acceleration_model_m_s2']['p95']/before['root_acceleration_model_m_s2']['p95']
        floor_limit=before['lowest_foot_abs_height_model_m']['p95']*1.1+.003
        gates={'rms_translation_at_most_limit':rms<=max_rms,
               'max_translation_at_most_0p25_model_m':float(magnitudes.max())<=.25,
               'p95_acceleration_reduced_at_least_20pct':accel_ratio<=.8,
               'floor_p95_not_materially_worse':after['lowest_foot_abs_height_model_m']['p95']<=floor_limit,
               'endpoint_vector_change_at_most_0p05_model_m':float(np.linalg.norm(endpoint_vector_change))<=.05}
        entry={'method':method,'accepted':all(gates.values()),'gates':gates,'motion':after,
               'translation_rms_model_m':rms,'translation_magnitude_model_m':summary(magnitudes),
               'p95_acceleration_ratio':accel_ratio,'endpoint_vector_change_model_m':endpoint_vector_change.tolist(),
               'endpoint_vector_change_norm_model_m':float(np.linalg.norm(endpoint_vector_change)),
               'first_last_sample_shift_model_m':np.linalg.norm(delta[[first,last]],axis=1).tolist(),
               'segments':blocks,'max_link_gap_s':max_gap}
        reports.append(entry)
        if entry['accepted']:accepted.append((after['root_acceleration_model_m_s2']['p95'],method,candidate,delta,entry))
    output.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(input_path.read_bytes()).hexdigest()
    report={'source_path':str(input_path.resolve()),'source_sha256':source_hash,'raw':before,
            'candidate_comparisons':reports,'selection_rule':'Lowest P95 root acceleration among bounded candidates; endpoint proximity is not optimized',
            'max_translation_rms_model_m':max_rms,'loop_constraint_applied':False,'contact_constraint_applied':False,
            'fixed_gauge_preserved':True,'timestamps_unchanged':True,'original_2d_preserved':True,
            'raw_input_modified':False,'metric_accuracy_claimed':False}
    if not accepted:
        report['status']='no_candidate_passed_gates';json_write(output/'temporal_report.json',report)
        return report
    _,method,candidate,delta,chosen=min(accepted,key=lambda x:x[0])
    result=dict(data)
    result.update(joints_world=candidate.astype(np.float32),world_joints=candidate.astype(np.float32),
                  joints_world_raw=raw.astype(np.float32),pelvis_world=candidate[:,[9,10]].mean(1).astype(np.float32),
                  pelvis_world_raw=root.astype(np.float32),root_translation_correction_world_m=delta.astype(np.float32),
                  temporal_root_method=np.array(method),source_world_sha256=np.array(source_hash),valid=valid)
    metadata=json.loads(str(data['metadata'])) if 'metadata' in data else {}
    metadata.update(temporal_root_processing={'method':method,'source_world_sha256':source_hash,
                                              'joint_relative_pose_changed':False,'loop_constraint':False,'contact_constraint':False,
                                              'original_2d_note':'Original SAM3D projection retained as source evidence; no claim that it reprojects the translated candidate'},
                    per_frame_correction_applied=True)
    result['metadata']=np.array(json.dumps(metadata,ensure_ascii=False))
    if 'source_metadata' not in result and 'metadata' in data:result['source_metadata']=data['metadata']
    relative_raw=raw-root[:,None]; relative_after=result['joints_world'].astype(float)-result['pelvis_world'].astype(float)[:,None]
    preservation_error=float(np.max(abs(relative_raw[valid]-relative_after[valid])))
    if preservation_error>2e-6:raise ValueError(f'Relative-pose preservation failed: {preservation_error}')
    temp=output/'world_joints.tmp.npz';np.savez_compressed(temp,**result);temp.replace(output/'world_joints.npz')
    closure=chosen['motion']['closure']
    closure.update(world_frame_id=str(data.get('world_frame_id','')),scale_status=str(data.get('metric_scale_status','model_prior_only')),
                   temporal_root_method=method,source_video_start_time_s=float(t[closure['start_selected_index']]),
                   source_video_end_time_s=float(t[closure['end_selected_index']]))
    json_write(output/'closure_metrics.json',closure)
    # Preserve the world basis, while making processing status explicit in the
    # copied gauge metadata as well as in the NPZ processing provenance.
    source_gauge=input_path.parent/'gauge.json'
    if source_gauge.exists():
        (output/'source_gauge.json').write_bytes(source_gauge.read_bytes())
        updated_gauge=json.loads(source_gauge.read_text(encoding='utf-8'))
        updated_gauge.update(per_frame_correction_applied=True,camera_gauge_unchanged=True,
                             temporal_root_method=method,per_frame_correction_applies_to='human_root_translation_only',
                             processing_report='temporal_report.json')
        json_write(output/'gauge.json',updated_gauge)
    report.update(status='complete_candidate',selected_method=method,selected=chosen,
                  relative_pose_max_abs_difference_model_m=preservation_error,
                  valid_frames=int(valid.sum()),selected_frames=len(t))
    json_write(output/'temporal_report.json',report)
    json_write(output/'status.json',{'status':'complete','candidate_kind':'temporal_root_only',
                                   'selected':len(t),'valid':int(valid.sum()),'loop_constraint_applied':False,
                                   'source_world_sha256':source_hash,'method':method})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-rms',type=float,default=.05)
    a=parser.parse_args();r=refine(a.input,a.output,a.max_rms)
    print(json.dumps({k:r[k] for k in ['status','selected_method','selected'] if k in r},indent=2))


if __name__=='__main__':main()
