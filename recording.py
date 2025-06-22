import time
from tqdm import tqdm
import os
import math

# Redefine method in capturing video to increase video capturing speed (add capture_only_step kwarg)
# Usage: MeshcatVisualizer.play = play
def play(self, q_trajectory, dt=None, callback=None, capture=False, capture_only_step=1, **kwargs):
    """
    Play a trajectory with given time step. Optionally capture RGB images and
    returns them.
    """
    nsteps = len(q_trajectory)
    if not capture:
        capture = self.has_video_writer()

    imgs = []
    for i in tqdm(range(nsteps)):
        t0 = time.time()
        self.display(q_trajectory[i])
        if callback is not None:
            callback(i, **kwargs)
        if capture and (capture_only_step < 1 or i % capture_only_step == 0):
            img_arr = self.captureImage()
            for _ in range(math.ceil(1 / capture_only_step)):
                if not self.has_video_writer():
                    imgs.append(img_arr)
                else:
                    self._video_writer.append_data(img_arr)
        t1 = time.time()
        elapsed_time = t1 - t0
        if dt is not None and elapsed_time < dt:
            self.sleep(dt - elapsed_time)
    if capture and not self.has_video_writer():
        return imgs


# Record a video based on already simulated data
def record(viz, qs, out_video_name: str, dtime: float, frameid: int, delay: float = 0):
    cos = input("Want to record a video? Enter to skip, number to create video with capture_only_step=<input>: ")
    if cos:
        try:
            cos = int(cos)
        except:
            try:
                cos = float(cos)
            except:
                print("You should input an integer! To avoid error, assuming integer 32")
                cos = 32
        try:
            with viz.create_video_ctx(f"out/{out_video_name}"):
                viz.play(qs, dtime, capture_only_step=cos)
        except Exception as e:
            print(f"Error while recording the video: {e}")

def get_out_video_name(filename: str) -> str:
    basename = os.path.splitext(os.path.basename(filename))[0]
    return f"showcase_{basename}.mp4"