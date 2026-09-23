import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from temporal_root import smooth_root, refine


class TemporalContract(unittest.TestCase):
    def test_never_bridge_invalid_or_time_gaps(self):
        t=np.arange(61)/15
        p=np.zeros((61,3));p[31:,0]=100;p[30]=np.nan
        valid=np.ones(61,bool);valid[30]=False
        for method in ['savgol5_poly3','savgol7_poly3','butterworth3hz_order2']:
            result,blocks,_=smooth_root(p,t,valid,method)
            np.testing.assert_allclose(result[valid],p[valid],atol=1e-10)
            self.assertTrue(np.isnan(result[30]).all());self.assertEqual(len(blocks),2)
        t[31:]+=2
        valid[30]=True;p[30]=0
        result,blocks,_=smooth_root(p,t,valid,'butterworth3hz_order2')
        np.testing.assert_allclose(result,p,atol=1e-10)
        self.assertEqual(len(blocks),2)

    def test_translation_only_and_source_contract(self):
        n=120;t=10+np.arange(n)/15
        joints=np.zeros((n,70,3))
        joints[:,:,2]=1.
        joints[:,[15,16,17,18,19,20],2]=0
        joints[:,:,0]=(np.linspace(0,2,n)+.04*np.sin(np.arange(n)*2))[:,None]
        joints[:,9,1]=-.1;joints[:,10,1]=.1
        uv=np.ones((n,70,2))*12
        valid=np.ones(n,bool)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            np.savez(p/'source.npz',joints_world=joints,world_joints=joints,valid=valid,timestamps_s=t,
                     keypoints_2d=uv,frame_indices=np.arange(n)*4+600,
                     metadata=np.array(json.dumps({'scale_status':'model_prior_only'})))
            report=refine(p/'source.npz',p/'candidate')
            self.assertEqual(report['status'],'complete_candidate')
            with np.load(p/'candidate/world_joints.npz') as z:
                np.testing.assert_array_equal(z['timestamps_s'],t)
                np.testing.assert_array_equal(z['keypoints_2d'],uv)
                np.testing.assert_allclose(z['joints_world_raw'],joints,atol=2e-7)
                delta=z['joints_world']-joints
                np.testing.assert_allclose(delta,np.broadcast_to(delta[:,0:1],delta.shape),atol=2e-7)
            self.assertFalse(report['loop_constraint_applied'])


if __name__=='__main__':unittest.main()
