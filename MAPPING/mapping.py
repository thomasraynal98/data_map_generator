import cv2
import pangolin
import OpenGL.GL as gl
import time
import numpy as np
import pywavefront
import argparse
import math as m

def pangolin_draw_coordinate():
    """
        DESCRIPTION: this function draw the world origine of the map.
    """
    gl.glLineWidth(3)
    gl.glColor3f(0, 0.0, 0.0)
    pangolin.DrawLines(
        [[0,0,0]], 
        [[1,0,0]], 3)   
    gl.glColor3f(0, 255, 0)
    pangolin.DrawLines(
        [[0,0,0]], 
        [[0,1,0]], 3)
    gl.glColor3f(0, 0, 255)
    pangolin.DrawLines(
        [[0,0,0]], 
        [[0,0,1]], 3)

def pangolin_init(w, h):
    """
        DESCRIPTION: init pangolin.
    """
    W, H = w, h
    pangolin.CreateWindowAndBind('Main', W, H)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(W, H, 420, 420, W // 2, H // 2, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, W/H)
    dcam.SetHandler(handler)

    # Create area to show video stream
    dimg = pangolin.Display('image')
    dimg.SetBounds(1.0, 0.66, 0, 0.33, W/H) # (debut hauteur, fin hauteur, debut largeur, fin largeur, ratio)
    dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

    return scam, dcam, dimg

def euler_to_rotation(r,p,y,t):
    """
        INPUT:
            * r, p, y = [1x1] euler element
            * t = [3x1] translation matrix
        OUTPUT:
            * pose = [3x4] tra
    """
    rz = np.array([[m.cos(y),-m.sin(y),0],
                   [m.sin(y),m.cos(y),0],
                   [0,0,1]])
    ry = np.array([[m.cos(p),0,m.sin(p)],
                   [0,1,0],
                   [-m.sin(p),0,m.cos(p)]])
    rx = np.array([[1,0,0],
                   [0,m.cos(r),-m.sin(r)],
                   [0,m.sin(r),m.cos(r)]])
    return np.concatenate((rz.dot(ry).dot(rx),t.reshape((3,1))),axis=-1)

def showing_mapping():
    """
        DESCRIPTION: Run a 3D mapping section and get live result.
            * first - only position
    """

    # READ ARG.
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("x")
    parser.add_argument("y")
    parser.add_argument("z")
    parser.add_argument("r")
    parser.add_argument("p")
    parser.add_argument("y")
    args = parser.parse_args()

    matrice = euler_to_rotation(float(args.r), float(args.p), float(args.y), np.array([args.x, args.y, args.z],dtype=float))

    # READ OBJ OBJECT.
    scene = pywavefront.Wavefront('/home/thomas/Documents/rane_slam/mk3slam.0.3/Final/data/{0}.obj'.format(args.path))
    points_3D = np.zeros((len(scene.vertices),3))
    for i in range(len(scene.vertices)):
        points_3D[i,:] = np.array([scene.vertices[i][0],scene.vertices[i][1],scene.vertices[i][2]])

    # PANGOLIN CONFIGURATION & INITIALISATION.
    w, h = 1280, 720
    scam, dcam, dimg = pangolin_init(w, h)
    texture = pangolin.GlTexture(
        w, h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
    )

    # ZED CONFIGURATION CAMERA.
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.coordinate_units = sl.UNIT.METER         # Set coordinate units
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    zed.open(init_params)

    # ZED CALIBRATION.
    zed.get_camera_information().camera_configuration.calibration_parameters.left_cam

    # INITIALISATION OBJECT FOR MAPPING.
    pymesh = sl.Mesh()        # Current incremental mesh.
    image = sl.Mat()          # Left image from camera.
    pose = sl.Pose()          # Pose object.

    # TRACKING PARAMETERS.
    tracking_parameters = sl.PositionalTrackingParameters()
    tracking_parameters.enable_area_memory = True
    tracking_parameters.area_file_path = '/home/thomas/Documents/rane_slam/mk3slam.0.3/Final/data/{0}.area'.format(args.path)

    # INITIALISATION POSITION

    t = sl.Transform()
    t[0,0] = matrice[0,0]
    t[1,0] = matrice[1,0]
    t[2,0] = matrice[2,0]
    t[0,1] = matrice[0,1]
    t[1,1] = matrice[1,1]
    t[2,1] = matrice[2,1]
    t[0,2] = matrice[0,2]
    t[1,2] = matrice[1,2]
    t[2,2] = matrice[2,2]
    t[0,3] = matrice[0,3]
    t[1,3] = matrice[1,3]
    t[2,3] = matrice[2,3]
  
    tracking_parameters.set_initial_world_transform(t)
    zed.enable_positional_tracking(tracking_parameters)

    # TIME PARAMETERS.
    last_call = time.time()
    runtime = sl.RuntimeParameters()

    while not pangolin.ShouldQuit():
        # -clear all.
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        # -draw coordinate origine.
        pangolin_draw_coordinate()

        # -get image.
        zed.grab(runtime)
        zed.retrieve_image(image, sl.VIEW.LEFT)

        # -get position and spatial mapping state.
        zed.get_position(pose)
        # Display the translation and timestamp
        py_translation = sl.Translation()
        tx = round(pose.get_translation(py_translation).get()[0], 3)
        ty = round(pose.get_translation(py_translation).get()[1], 3)
        tz = round(pose.get_translation(py_translation).get()[2], 3)
        print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, pose.timestamp.get_milliseconds()))

        # Display the orientation quaternion
        py_orientation = sl.Orientation()
        ox = round(pose.get_orientation(py_orientation).get()[0], 3)
        oy = round(pose.get_orientation(py_orientation).get()[1], 3)
        oz = round(pose.get_orientation(py_orientation).get()[2], 3)
        ow = round(pose.get_orientation(py_orientation).get()[3], 3)
        #print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

        # DRAW ALL STATE.
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)

        # -transform brute data in numpy to draw pose.
        a = pose.pose_data()
        pose2 = np.array([[a[0,0],a[0,1],a[0,2],a[0,3]],
                          [a[1,0],a[1,1],a[1,2],a[1,3]],
                          [a[2,0],a[2,1],a[2,2],a[2,3]],
                          [a[3,0],a[3,1],a[3,2],a[3,3]]])
        print(pose2)
        pangolin.DrawCamera(pose2, 0.5, 0.75, 0.8)

        # -draw all points on the map.
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(points_3D)    

        # DISPLAY CAMERA VIDEO.
        img = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2RGB)
        img = cv2.flip(img, 0)
        texture.Upload(img, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        dimg.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        texture.RenderToViewport()

        # END OF CYCLE.
        pangolin.FinishFrame()
    
    #sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
    zed.disable_positional_tracking()
    zed.close()

if __name__=="__main__":
    showing_mapping()