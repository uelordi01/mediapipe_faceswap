import numpy as np
import dlib
import cv2
class DLIBFaceswaper:
  def configure_recording(self,  width, height, out_filename):
      self.out = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))
  def write_record(self, img):
      self.out.write(img);
  def __init__(self):
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

  def updateBaseImg(self, img_path):
      self.base_img = cv2.imread(img_path)
      if self.base_img is None:
          print("BASE IMAGE NOT FOUND, please check if the specified image is in this folder")
          assert(False)
      self.base_img_gray = cv2.cvtColor(self.base_img, cv2.COLOR_BGR2GRAY)
      self.mask = np.zeros_like(self.base_img_gray)
      self.base_face_handler = self.extract_landmarks(self.base_img, self.base_img_gray)
      self.triangle_handler = self.triangulate_faces(self.base_face_handler["landmarks"],  self.base_img)
      return self.base_img
  def draw_subimage(self, sub_image, image):
      image_aspect_ratio = sub_image.shape[0] / sub_image.shape[1]
      desired_width = int(image.shape[0] / 3)
      desired_height = int(desired_width * image_aspect_ratio)
      size = (desired_width, desired_height)
      resized_Image = cv2.resize(sub_image, size)
      image[0:desired_height, 0:desired_width] = resized_Image
      return image
  def process_target(self, img):
      img2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      self.img_target = img.copy()
      self.target_face_handler = self.extract_landmarks(img, img2_gray)
      self.img2_new_face = np.zeros_like(img)
      if len(self.target_face_handler["landmarks"]) > 0:
          points2 = np.array(self.target_face_handler["landmarks"], np.int32)
          convexhull2 = cv2.convexHull(points2)

          for triangle_index in self.triangle_handler["index"]:
              # Triangulation of the first face
              tpt1 = self.base_face_handler["landmarks"][triangle_index[0]]
              tpt2 = self.base_face_handler["landmarks"][triangle_index[1]]
              tpt3 = self.base_face_handler["landmarks"][triangle_index[2]]
              triangle1 = np.array([tpt1, tpt2, tpt3], np.int32)

              rect1 = cv2.boundingRect(triangle1)
              (x, y, w, h) = rect1
              cropped_triangle = self.base_img[y: y + h, x: x + w]
              cropped_tr1_mask = np.zeros((h, w), np.uint8)

              points = np.array([[tpt1[0] - x, tpt1[1] - y],
                                 [tpt2[0] - x, tpt2[1] - y],
                                 [tpt3[0] - x, tpt3[1] - y]], np.int32)

              cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

              # Triangulation of second face
              t2pt1 = self.target_face_handler["landmarks"][triangle_index[0]]
              t2pt2 = self.target_face_handler["landmarks"][triangle_index[1]]
              t2pt3 = self.target_face_handler["landmarks"][triangle_index[2]]
              triangle2 = np.array([t2pt1, t2pt2, t2pt3], np.int32)

              rect2 = cv2.boundingRect(triangle2)
              (x, y, w, h) = rect2

              cropped_tr2_mask = np.zeros((h, w), np.uint8)

              points2 = np.array([[t2pt1[0] - x, t2pt1[1] - y],
                                  [t2pt2[0] - x, t2pt2[1] - y],
                                  [t2pt3[0] - x, t2pt3[1] - y]], np.int32)

              cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

              # Warp triangles
              points = np.float32(points)
              points2 = np.float32(points2)
              M = cv2.getAffineTransform(points, points2)
              warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
              warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

              # Reconstructing destination face
              img2_new_face_rect_area = self.img2_new_face[y: y + h, x: x + w]
              img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
              _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
              warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

              img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
              self.img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
          # Face swapped (putting 1st face into 2nd face)
          img2_face_mask = np.zeros_like(img2_gray)
          img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
          # cv2.imshow("pabit", img2_head_mask)
          img2_face_mask = cv2.bitwise_not(img2_head_mask)
          seam_clone = img.copy()
          self.img2_head_noface = cv2.bitwise_and(seam_clone, seam_clone, mask=img2_face_mask)

          cv2.imshow("no_Head", self.img2_head_noface)
          self.result = cv2.add(self.img2_head_noface, self.img2_new_face)

          (x, y, w, h) = cv2.boundingRect(convexhull2)
          center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

          self.seamlessclone = cv2.seamlessClone(self.result, seam_clone,
                                            img2_head_mask, center_face2, cv2.MIXED_CLONE)
  def extract_index_nparray(self, nparray):
      index = None
      for num in nparray[0]:
          index = num
          break
      return index
  def extract_landmarks(self, img, img_gray):
      faces = self.detector(img_gray)
      landmarks_points = []
      for face in faces:
          landmarks = self.predictor(img_gray, face)
          landmarks_points = []
          for n in range(0, 68):
              x = landmarks.part(n).x
              y = landmarks.part(n).y
              landmarks_points.append((x, y))
      return {"landmarks": landmarks_points, "img": img}

  def triangulate_faces(self, landmarks_points, img, draw=False):
      points = np.array(landmarks_points, np.int32)
      convexhull = cv2.convexHull(points)
      cv2.fillConvexPoly(self.mask, convexhull, 255)
      rect = cv2.boundingRect(convexhull)
      subdiv = cv2.Subdiv2D(rect)
      subdiv.insert(landmarks_points)
      triangles = subdiv.getTriangleList()
      triangles = np.array(triangles, dtype=np.int32)
      if draw:
          img = self.draw_mask(triangles, img)
      indexes_triangles = []
      for t in triangles:
          pt1 = (t[0], t[1])
          pt2 = (t[2], t[3])
          pt3 = (t[4], t[5])

          index_pt1 = np.where((points == pt1).all(axis=1))
          index_pt1 = self.extract_index_nparray(index_pt1)

          index_pt2 = np.where((points == pt2).all(axis=1))
          index_pt2 = self.extract_index_nparray(index_pt2)

          index_pt3 = np.where((points == pt3).all(axis=1))
          index_pt3 = self.extract_index_nparray(index_pt3)

          if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
              triangle = [index_pt1, index_pt2, index_pt3]
              indexes_triangles.append(triangle)
      return {"triangles": triangles, "index": indexes_triangles, "img": img}

  def draw_mask(triangle_list, img):
      for triangle in triangle_list:
          cv2.line(img, (triangle[0], triangle[1]),
                   (triangle[2], triangle[3]), (255, 0, 0), 5)
          cv2.line(img, (triangle[2], triangle[3]),
                   (triangle[4], triangle[5]), (255, 0, 0), 5)
          cv2.line(img, (triangle[4], triangle[5]),
                   (triangle[0], triangle[1]), (255, 0, 0), 5)
      return img
  def draw_base_landmarks(self):
      return self.draw_landmarks(self.base_face_handler["landmarks"], self.base_img)
  def draw_base_triangles(self):
      return self.draw_mask(self.triangle_handler["triangles"], self.base_img)
  def draw_target_landmarks(self):
      return self.draw_landmarks(self.target_face_handler["landmarks"], self.img_target)
  def draw_target_triangles(self):
      return self.draw_mask(self.triangle_handler["triangles"], self.base_img)
  def draw_landmarks(self, landmarks, img):
      draw_img = img.copy()
      radius = 3
      color = (0, 255, 0)
      thickness = -1
      for land in landmarks:
          img = cv2.circle(draw_img,
                           land,
                           radius,
                           color,
                           thickness)
      return draw_img
  def show_seamlessclone(self):
      if len(self.target_face_handler["landmarks"]) > 0:
        self.seamlessclone = self.draw_subimage(self.base_img, self.seamlessclone)
        cv2.imshow("clone", self.seamlessclone)
  def show_result(self):
      if len(self.target_face_handler["landmarks"]) > 0:
        cv2.imshow("result", self.result)
  def getSeamlessIMG(self):
      return self.seamlessclone
  def getResultImg(self):
      return self.result
  def getResultMask(self):
      return self.img2_head_noface



