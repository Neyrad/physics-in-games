import time
import numpy as np
import pinocchio as pin
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
from tqdm import tqdm

from recording import play, record, get_out_video_name
MeshcatVisualizer.play = play

def sym_loop():
    robot = example_robot_data.load("double_pendulum")
    model = robot.model
    data = model.createData()
    print(list(frame.name for frame in model.frames))

    q = np.array([0.5, -0.5])
    q_target = np.array([0.0, 0.0])
    v = np.zeros(model.nv)

    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.display(q)

    k = 20.0
    dt = 0.01
    total_time = 10.0
    n_steps = int(total_time / dt)

    damping = 1.0

    qs = [q]
    vs = [v]
    for _ in tqdm(range(n_steps)):
        pin.forwardKinematics(model, data, q)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        f_ext = [pin.Force.Zero() for _ in range(model.njoints)]
        for j in range(1, model.njoints):
            com = pin.centerOfMass(model, data, q, j)
            com_target = pin.centerOfMass(model, data, q_target, j)
            force = -k * (com - com_target)
            f_ext[j] = pin.Force(force, np.zeros(3))

        tau_g = -pin.rnea(model, data, q, np.zeros_like(v), np.zeros_like(v))
        a = pin.aba(model, data, q, v, tau_g, f_ext)

        v += a * dt
        v *= np.exp(-damping * dt)
        q = pin.integrate(model, q, v * dt)

        qs.append(q.copy())
        vs.append(v.copy())
        viz.display(q)

        time.sleep(dt)

    OUT_VIDEO_NAME = get_out_video_name(__file__)
    record(viz, qs, OUT_VIDEO_NAME, dt, model.getFrameId("base_link"))

if __name__ == "__main__":
    sym_loop()
