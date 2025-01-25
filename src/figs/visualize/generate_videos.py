import imageio
import numpy as np

def images_to_mp4(images:np.ndarray, filename:str, fps:int):
    """
    Convert an array of images (B, H, W, C) in uint8 format into an MP4 video using imageio.

    Args:
        - images:   A numpy array of shape (B, H, W, C) with uint8 type.
        - filename: The output MP4 filename.
        - fps:      Frames per second for the video.
    """
    # Validate input
    assert len(images.shape) == 4, "Input array must have shape (B, H, W, C)"
    assert images.dtype == np.uint8, "Images must be of type uint8"
    assert images.shape[-1] in [1, 3], "Images must have 1 (grayscale) or 3 (RGB) channels"

    # Convert grayscale to RGB if necessary
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)

    # Write video using imageio
    with imageio.get_writer(filename, format='FFMPEG', mode='I', fps=fps, macro_block_size=None) as writer:
        for frame in images:
            writer.append_data(frame)