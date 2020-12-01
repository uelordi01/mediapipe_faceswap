import cv2
import dlib_faceswap


def init_windows():
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', width, height)
    cv2.namedWindow('clone', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('clone', width, height)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', width, height)


indexes_triangles = []

flag_draw_landmarks = False
flag_draw_mask = False
flag_change_face = False
face_list_index = 0
change_face_list = []  # write here your reference photos.
MAX_FACESWAP_IMAGES = len(change_face_list) - 1
if len(change_face_list) == 0:
    print("WARNING: (EMPTY IMAGE ARRAY) PLEASE PUT IN THE variable change_face_list the images that you would like to "
          "swap")
    assert False
width = 640
height = 480

camera_opened = False
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if cap is None or not cap.isOpened():
    print('Warning: unable to open camera: ')
else:
    camera_opened = True

if camera_opened:
    init_windows()
    dlib_handler = dlib_faceswap.DLIBFaceswaper()
    base_img = dlib_handler.updateBaseImg(change_face_list[face_list_index])

    while True:
        ret, img_target = cap.read()
        if img_target is not None:
            has_captured_camera_landmarks = False
            if flag_change_face:
                base_img = dlib_handler.updateBaseImg(change_face_list[face_list_index])
                flag_change_face = False
            input_image = base_img.copy()

            if flag_draw_mask:
                input_image = dlib_handler.draw_base_triangles()

            dlib_handler.process_target(img_target)
            if flag_draw_landmarks:
                input_image = dlib_handler.draw_base_landmarks()
                img_target = dlib_handler.draw_target_landmarks()
            cv2.imshow("img2", img_target)
            dlib_handler.show_seamlessclone()
            dlib_handler.show_result()
            key = cv2.waitKey(30)
            if key == 27:
                break
            if key == 49:
                flag_draw_landmarks = not flag_draw_landmarks
            if key == 50:
                flag_draw_mask = not flag_draw_mask
            if key == 99:
                flag_change_face = not flag_change_face
                face_list_index = face_list_index + 1
                if face_list_index > MAX_FACESWAP_IMAGES:
                    face_list_index = 0
            if key == 83 or key == 115:
                break
                # cv2.imwrite("img2.jpg", img_target)
                # cv2.imwrite("clone.jpg", dlib_handler.getSeamlessIMG())
                # cv2.imwrite("result.jpg", dlib_handler.getResultImg())

    cap.release()
    cv2.destroyAllWindows(input_image)
