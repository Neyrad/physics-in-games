import time
import numpy as np
import pinocchio as pin
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
from tqdm import tqdm
from scipy.optimize import minimize
import math

from recording import play, record, get_out_video_name
MeshcatVisualizer.play = play

class RotationBoth:
    def __init__(self, rotations: tuple[int, int]):
        self.a, self.b = rotations
        self.rot1 = self.a % (2 * np.pi)
        if self.rot1 > np.pi:
            self.rot1 -= 2 * np.pi
        self.rot2 = (self.a + self.b) % (2 * np.pi)
        if self.rot2 > np.pi:
            self.rot2 -= 2 * np.pi

    def modify(self, current: tuple[int, int]):
        ROT = 0.2
        value = min(ROT, abs(current[0] + current[1]) * 0.2)
        self.rot2 -= (value / ROT) * ROT * np.sign(current[0])
        return self

    def __sub__(self, oth):
        result = np.array([self.rot1 - oth.rot1, self.rot2 - oth.rot2])
        for i in range(2):
            result[i] = result[i] % (2 * np.pi)
            if result[i] >= np.pi:
                result[i] = result[i] - 2 * np.pi
        result[0] = (result[1] / np.pi) ** 3 * np.pi
        return result

class PID:
    def __init__(self, target: np.ndarray):
        self.target = target
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, current: np.ndarray, velocity: np.ndarray, dt: float):
        MAX_POWER = 10
        KP = np.array([100.0, 100.0])
        KI = np.array([-0.01, 0.01])
        KD = np.array([0.08, 0.8])
        KD_MAX_MOD = 500
        VELOCITY_STEP = dt * 5
        K = 0.1
        FIRST_WEIGHT = 0.2

        current = current + velocity * VELOCITY_STEP
        error = RotationBoth(self.target).modify(current) - RotationBoth(current)
        error -= velocity * 0.02
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        result = KP * error + KI * self.integral + np.clip(-KD * derivative, -KD_MAX_MOD, KD_MAX_MOD)
        result *= K
        result[0] = min(MAX_POWER, max(-MAX_POWER, -result[0] * FIRST_WEIGHT + result[1])) * -1
        result[1] = 0
        return result

    def __call__(self, qs: np.ndarray, vs: np.ndarray, torque0: np.ndarray, dt: float):
        return self.compute(qs[-1], vs[-1], dt)

def optimize_torque(tau_init, q, v, model, data):
    def objective(tau):
        return 0.5 * np.dot(tau, tau)
    bounds = [(-10.0, 10.0)] * model.nv
    result = minimize(objective, tau_init, bounds=bounds)
    return result.x if result.success else tau_init

def sim_loop():
    robot = example_robot_data.load('double_pendulum')
    model = robot.model
    data = model.createData()
    print(list(frame.name for frame in model.frames))

    viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    START_POSITION = np.array([0.0, 0.0])
    START_VELOCITY = np.array([1.0, 0.0])
    TARGET_POSITION = np.array([0.0, 0.0])
    dt = 0.01
    NSTEPS = 1500
    FRAMERATE = 60
    FRICTION = 0.02

    pid = PID(target=TARGET_POSITION)
    torque0 = np.zeros(model.nv)

    q = START_POSITION
    v = START_VELOCITY
    qs = [q]
    vs = [v]

    viz.display(q)
    time.sleep(1)

    for _ in tqdm(range(NSTEPS)):
        def xdot(qc, vc, torquec):
            friction = torquec - FRICTION * vc
            return vc, pin.aba(model, data, qc, vc, friction)

        k1 = xdot(q,                  v,                  torque0)
        k2 = xdot(q + dt/2 * k1[0],   v + dt/2 * k1[1],   torque0)
        k3 = xdot(q + dt/2 * k2[0],   v + dt/2 * k2[1],   torque0)
        k4 = xdot(q + dt * k3[0],     v + dt * k3[1],     torque0)

        q += dt / 6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        v += dt / 6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

        pin.computeAllTerms(model, data, q, v)

        torque_pid = pid(qs, vs, torque0, dt)
        torque0 = optimize_torque(torque_pid, q, v, model, data)

        qs.append(q.copy())
        vs.append(v.copy())
        viz.display(q)
        time.sleep(dt)

    OUT_VIDEO_NAME = get_out_video_name(__file__)
    print(qs)
    record(viz, qs, OUT_VIDEO_NAME, dt, model.getFrameId("base_link"))

if __name__ == "__main__":
    sim_loop()
