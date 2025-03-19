"""
The input for all classic algorithms in this folder is
the target reconstructed image a_target (NCHW), N images of size HW with C channels, generally 1*3*HW or 1*1*HW
the target reconstructed depth z (NDCHW), for N different images there are D reconstructions at different depths, distinguishing channel C, generally 1*1*1*1*1 or 1*1*1*1*1
"""