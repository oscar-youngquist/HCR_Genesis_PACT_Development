"""Opt-in headless RGB recorder. No physics steps, random draws or training imports."""
import time
import numpy as np


class Recorder:
    def __init__(self,o):
        import imageio.v2 as imageio
        import omni.replicator.core as rep
        from pxr import UsdGeom
        from scripts.eval_hard_pact_tradeoff import write_json
        self.o=o;self.seconds=0.;self.frames=0;self.cameras=[];self.writers=[]
        # Positive-height pyramid stairs spawn on the HIGH center platform:
        # outward locomotion descends. The legacy family label is stairs_up.
        self.ids=[i for i,p in enumerate(o.manifest['target_plan']) if p['family']=='stairs_up'][:3]
        if len(self.ids)!=3:raise ValueError('Three stair-down scenarios required')
        o.s._build_lights()
        for i in self.ids:
            camera=UsdGeom.Camera.Define(o.s._stage,f'/World/EvaluationCamera{i}')
            camera.GetFocalLengthAttr().Set(22.)
            camera.GetHorizontalApertureAttr().Set(24.)
            camera.GetClippingRangeAttr().Set((.05,1000.))
            product=rep.create.render_product(camera.GetPath().pathString,(640,480))
            annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach([product])
            self.cameras.append((camera,annotator,product))
            self.writers.append(None if getattr(o.args,'discard_video',False) else
                imageio.get_writer(str(o.out/f'stairs_down_env{i}.mp4'),fps=50,
                codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-threads','1']))
        self.position()
        # Render-only warmup: never advance physics or policy state.
        start=time.monotonic()
        for _ in range(15):o.s._sim.render()
        self.seconds+=time.monotonic()-start
        write_json(o.out/'video_manifest.json',dict(env_ids=self.ids,fps=50,playback_speed=1.,
            physics_dt=.005,control_dt=.02,frame_time='post-physics/pre-reset, (control_index+1)*dt',
            view='world-up Z; fixed world-axis three-quarter offset following translation, not attitude',
            terrain_note='Legacy stairs_up family: positive pyramid, high central spawn; outward travel descends. Verify height change; no scenario/command override.',
            height='base-link z minus mean of the simulator terrain-height samples; exact termination quantity',
            thresholds={k:getattr(o.e.cfg.termination,k) for k in ('roll_threshold','pitch_threshold','height_min','height_max')},
            no_extra_physics_steps=True,frames_discarded=getattr(o.args,'discard_video',False)))

    def position(self):
        from pxr import Gf,UsdGeom
        positions=self.o.s._robot.data.root_link_pos_w[self.ids].detach().cpu().numpy()
        for p,(cam,_,_) in zip(positions,self.cameras):
            eye=p+np.array([1.8,-2.,1.3]);target=p+np.array([.25,0.,-.08])
            matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
            UsdGeom.Xformable(cam).MakeMatrixXform().Set(matrix)

    def capture(self):
        from PIL import Image,ImageDraw
        o=self.o;start=time.monotonic();before=o.s._sim.current_time
        self.position();o.s._sim.render()
        assert o.s._sim.current_time==before,'Rendering advanced simulation time'
        thresholds=o.e.cfg.termination
        for i,(_,annotator,_),writer in zip(self.ids,self.cameras,self.writers):
            frame=np.asarray(annotator.get_data())
            if frame.shape[:2]!=(480,640):raise RuntimeError(f'Unavailable camera frame: {frame.shape}')
            image=Image.fromarray(frame[...,:3].copy());draw=ImageDraw.Draw(image)
            get=lambda k:o.controls[k][o.t,i].detach().cpu().numpy()
            roll,pitch=get('tilt');height=float(get('base_height'))
            velocity=o.body_velocity()[i,[0,1,5]].detach().cpu().numpy()
            command=o.command[i].detach().cpu().numpy()
            corr=o.arrays['correction'][o.t*4+3,i].detach().cpu().numpy()
            clipped=bool(o.arrays.get('clipped',np.zeros((o.args.steps*4,10)))[o.t*4+3,i])
            reason=get('reasons');reset=bool(get('reset'))
            names=['contact','roll','pitch','height_low','height_high','timeout','terrain_exit']
            text=[f'{o.args.worker} | env {i} | t={(o.t+1)*o.e.dt:.2f}s | 1x playback',
                f'disturbance={bool(o.e._current_sustained_active_mask[i])} roll={roll:+.3f}/{thresholds.roll_threshold} pitch={pitch:+.3f}/{thresholds.pitch_threshold}',
                f'termination height={height:.3f}m bounds=[{thresholds.height_min},{thresholds.height_max}]',
                f'cmd vx/vy/wz={command.round(2)} measured={velocity.round(2)}',
                f'correction RMS={np.sqrt(np.mean(corr*corr)):.2f}Nm clip={clipped}',
                f'bounds={",".join(n for n,b in zip(names,reason) if b) or "none"} ORIGINAL_FAIL={bool(get("failure"))} RESET={reset}']
            if 'probe_active' in o.controls:text.append(f'probe={bool(get("probe_active"))} outcome={int(get("probe_outcome"))}')
            draw.rectangle((0,0,639,16*len(text)+8),fill=(12,12,12))
            for j,line in enumerate(text):draw.text((5,4+j*16),line,fill=(255,230,170) if reset else 'white')
            if writer is not None:writer.append_data(np.asarray(image))
        self.frames+=1;self.seconds+=time.monotonic()-start

    def close(self):
        from scripts.eval_hard_pact_tradeoff import write_json
        for writer in self.writers:
            if writer is not None:writer.close()
        write_json(self.o.out/'video_timing.json',dict(frames=self.frames,render_encode_wall_s=self.seconds,
            simulation_seconds=self.frames*self.o.e.dt,collection_includes_rendering=True,physics_time_unchanged=True))
