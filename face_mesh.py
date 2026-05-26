import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    circle_radius=1
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    # Read frame
    success, frame = cap.read()

    if not success:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    # Draw landmarks if face detected
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Display frame
    cv2.imshow("Face Mesh", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # Drawing utility
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(
#     thickness=1,
#     circle_radius=1
# )

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     success, frame = cap.read()

#     if not success:
#         break

#     # Flip image for mirror effect
#     frame = cv2.flip(frame, 1)

#     # Convert BGR to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process frame
#     results = face_mesh.process(rgb_frame)

#     # If face detected
#     if results.multi_face_landmarks:

#         for face_landmarks in results.multi_face_landmarks:

#             # Draw landmarks
#             mp_drawing.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=drawing_spec,
#                 connection_drawing_spec=drawing_spec
#             )

#     # Show frame
#     cv2.imshow("Face Mesh", frame)

#     # Press q to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
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