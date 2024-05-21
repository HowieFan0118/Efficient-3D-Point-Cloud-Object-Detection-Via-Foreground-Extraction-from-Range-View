# The Code for RIESS Module
The RIESS module performs foreground extraction based on semantic segmentation on the range image with RIE channels obtained from the upstream module, providing pixel-level label classification.

Since the categories for 3D point cloud object detection are vehicles, pedestrians, and bicycles, other background points in the environment are relatively useless. Therefore, our method considers the points within the 3D bounding boxes of categories mentioned above in the original point cloud as foreground points, and the rest as background points. Thus, this module applies a binary semantic segmentation task to the RIE image.
