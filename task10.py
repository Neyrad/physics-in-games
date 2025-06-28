import numpy as np
import pinocchio as pin
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
from tqdm import tqdm
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from recording import play, record, get_out_video_name
MeshcatVisualizer.play = play

robot = example_robot_data.load('double_pendulum')
model = robot.model
data = model.createData()

viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

q = np.array([np.pi, 0.0])
v = np.array([-14.0, -10.0])

qs = [q.copy()]
viz.display(q)
time.sleep(1)

q_stable = np.array([np.pi, 0.0])
pin.computeAllTerms(model, data, q_stable, np.zeros(model.nv))
U_ref = pin.computePotentialEnergy(model, data, q_stable)

dt = 0.001
T = 1
N = int(T / dt)
q_target = np.array([0.0, 0.0])
integral_error = np.zeros(model.nq)
energies = []

def compute_energy_normalized(model, data, q, v, U_ref):
    pin.computeAllTerms(model, data, q, v)
    kinetic = 0.5 * v @ data.M @ v
    potential = pin.computePotentialEnergy(model, data, q) - U_ref
    return kinetic + potential

def compute_energy_guided_pid_accel(model, data, q, v, dt, q_target):
    Kp = 100.0
    Ki = 0.0
    Kd = 20.0

    error = q_target - q

    global integral_error
    integral_error += error * dt

    a_target = Kp * error + Ki * integral_error - Kd * v

    def objective(a_flat):
        a = np.array(a_flat)
        v_next = v + a * dt
        q_step = v * dt + 0.5 * a * dt**2
        q_next = pin.integrate(model, q, q_step)

        pin.computeAllTerms(model, data, q_next, v_next)
        M_next = data.M

        kinetic = 0.5 * v_next @ M_next @ v_next
        potential = pin.computePotentialEnergy(model, data, q_next)
        total_energy = kinetic + potential

        return total_energy

    a0 = a_target
    bounds = [(-50, 50)] * model.nv

    res = minimize(objective, a0, bounds=bounds, method='SLSQP')

    if res.success:
        return res.x
    else:
        print("ERROR: cannot find min, returning a_target")
        return a_target

def compute_accel(q, v):
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("ERROR: compute_accel(): unstable input")
    pin.computeAllTerms(model, data, q, v)
    a = compute_energy_guided_pid_accel(model, data, q, v, dt, q_target)
    return a

def plot_energy(energies, dt):
    times = [i * dt for i in range(len(energies))]
    plt.figure(figsize=(8,5))
    plt.plot(times, energies, label='Full mechanical energy')
    plt.xlabel('Time, s')
    plt.ylabel('Energy')
    plt.title('Full mechanical energy dynamics')
    plt.grid(True)
    plt.legend()
    plt.show()

for i in tqdm(range(N)):
    a1 = compute_accel(q, v)
    dq1 = v * dt
    dv1 = a1 * dt

    v2 = v + dv1 / 2
    q2 = pin.integrate(model, q, dq1 / 2)
    a2 = compute_accel(q2, v2)
    dq2 = v2 * dt
    dv2 = a2 * dt

    v3 = v + dv2 / 2
    q3 = pin.integrate(model, q, dq2 / 2)
    a3 = compute_accel(q3, v3)
    dq3 = v3 * dt
    dv3 = a3 * dt

    v4 = v + dv3
    q4 = pin.integrate(model, q, dq3)
    a4 = compute_accel(q4, v4)
    dq4 = v4 * dt
    dv4 = a4 * dt

    dq = (dq1 + 2 * dq2 + 2 * dq3 + dq4) / 6
    dv = (dv1 + 2 * dv2 + 2 * dv3 + dv4) / 6

    v += dv
    q = pin.integrate(model, q, dq)

    if not np.isfinite(q).all() or not np.isfinite(v).all():
        print(f"ERROR: unstable data on step {i}")
        break

    qs.append(q.copy())
    viz.display(q)

    energy = compute_energy_normalized(model, data, q, v, U_ref)
    energies.append(energy)

    time.sleep(dt)

plot_energy(energies, dt)

OUT_VIDEO_NAME = get_out_video_name(__file__)
record(viz, qs, OUT_VIDEO_NAME, dt, model.getFrameId("base_link"))
