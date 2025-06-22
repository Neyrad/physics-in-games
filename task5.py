import time
import numpy as np
import pinocchio as pin
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
from tqdm import tqdm
from meshcat.geometry import Box, Sphere, MeshLambertMaterial
from meshcat.transformations import translation_matrix

from recording import play, record, get_out_video_name
MeshcatVisualizer.play = play

FLOOR_HEIGHT = -0.23
LENGTH2 = 0.21
COLLISION_SENSITIVITY = 700

def sym_loop():
    robot = example_robot_data.load("double_pendulum")
    model = robot.model
    data = model.createData()

    print([frame.name for frame in model.frames])

    q = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])

    dt = 0.01
    damping = 0.05
    n_steps = 500

    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.display(q)

    qs = [q]
    vs = [v]

    frame_id = model.getFrameId("link2")
    started_colliding = False
    v_sign = 1

    for _ in tqdm(range(n_steps)):
        oMf = data.oMf[frame_id]
        end_local = np.array([0, 0, LENGTH2, 1.0])
        end_world = oMf @ end_local

        tip_transform = np.eye(4)
        tip_transform[:3, 3] = end_world[:3]
        viz.viewer["tip"].set_transform(tip_transform)

        floor_size = [2.0, 2.0, 0.01]
        floor_transform = translation_matrix([0.0, 0.0, FLOOR_HEIGHT - 0.01])
        viz.viewer["floor"].set_object(Box(floor_size), MeshLambertMaterial(color=0x444444))
        viz.viewer["floor"].set_transform(floor_transform)

        tau_g = -pin.rnea(model, data, q, np.zeros_like(v), np.zeros_like(v))
        M = pin.crba(model, data, q)
        a = np.linalg.solve(M, tau_g - damping * v)

        if end_world[2] < FLOOR_HEIGHT:
            if not started_colliding:
                started_colliding = True
                depth = abs(end_world[2] - FLOOR_HEIGHT)
                v_sign = 1 if (v[0] + v[1] < 0) else -1
            v[1] = v_sign * depth * COLLISION_SENSITIVITY
        elif end_world[2] > FLOOR_HEIGHT + 0.01:
            started_colliding = False

        v += a * dt
        q = pin.integrate(model, q, v * dt)

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        qs.append(q.copy())
        vs.append(v.copy())

        viz.display(q)
        time.sleep(dt)

    out_video_name = get_out_video_name(__file__)
    record(viz, qs, out_video_name, dt, frame_id)


if __name__ == "__main__":
    sym_loop()
