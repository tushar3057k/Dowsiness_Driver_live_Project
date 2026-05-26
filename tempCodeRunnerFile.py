import cv2
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import mediapipe as mp

# # Base options
# base_options = python.BaseOptions(
#     model_asset_path=None
# )

# # Face Landmarker options
# options = vision.FaceLandmarkerOptions(
#     base_options=base_options,
#     output_face_blendshapes=False,
#     output_facial_transformation_matrixes=False,
#     num_faces=1
# )

# # Create landmarker
# detector = vision.FaceLandmarker.create_from_options(options)

# # Webcam
# cap = cv2.VideoCapture(0)

# while True:
#     success, frame = cap.read()

#     if not success:
#         break

#     frame = cv2.flip(frame, 1)

#     cv2.imshow("Webcam", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()