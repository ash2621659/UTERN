import mediapipe as mp
import cv2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import sys
import os

def set_parameter():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(color=(240,0,0), thickness=1, circle_radius=1)

    return mp_drawing, mp_drawing_styles, mp_face_mesh, drawing_spec

def setup_landmarks(face_landmarks, image, mp_drawing, mp_face_mesh, mp_drawing_styles):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(240,0,0), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=None,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(240,0,0), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style()) #mp_face_mesh.FACEMESH_CONTOURS
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=None,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(240,0,0), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()) #mp_face_mesh.FACEMESH_IRISES


def find_points(image_path):
    image = cv2.imread(image_path)
    mp_drawing, mp_drawing_styles, mp_face_mesh, drawing_spec = set_parameter()
    with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            #print(len(results.multi_face_landmarks))
            for face_landmarks in results.multi_face_landmarks:
                fl = [face_landmarks]
                #print(len(fl))
                setup_landmarks(face_landmarks, image, mp_drawing, mp_face_mesh, mp_drawing_styles)
                connections1 = mp_face_mesh.FACEMESH_TESSELATION
            fl_cord = results.multi_face_landmarks[0].landmark
            connections1 = list(connections1)
            cord_list = []
            for idx in fl_cord:
                cords = _normalized_to_pixel_coordinates(idx.x,idx.y,image.shape[1],image.shape[0])
                cord_list.append(cords)
                
    return cord_list , connections1

if __name__=="__main__":
    image_path = sys.argv[1]
    outpath = sys.argv[2]
    conn_outpath = sys.argv[3]
    print(image_path)
    print(outpath)
    cord_list, connections = find_points(image_path)
    cord_list = [str(cord) for cord in cord_list]
    cord_text = ("\n").join(cord_list)

    connections = [str(conn) for conn in connections]
    conn_text = ("\n").join(connections)

    #saving file
    file1 = open(outpath, "w")
    file1.writelines(cord_text)
    file1.close() 

    #saving file
    file1 = open(conn_outpath, "w")
    file1.writelines(conn_text)
    file1.close() 

            