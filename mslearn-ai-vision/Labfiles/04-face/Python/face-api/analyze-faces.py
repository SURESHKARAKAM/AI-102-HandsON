from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection03
from azure.core.credentials import AzureKeyCredential

def main():
    global face_client

    try:
        # Get Configuration Settings
        load_dotenv()
        cog_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        cog_key = os.getenv('AI_SERVICE_KEY')

        # Authenticate Face client
        face_client = FaceClient(
            endpoint=cog_endpoint,
            credential=AzureKeyCredential(cog_key)
        )

        # Menu for face functions
        print('1: Detect faces in a single image')
        print('2: Detect faces in multiple images from a folder')
        print('Any other key to quit')

        command = input('Enter a number: ')
        if command == '1':
            DetectFaces(os.path.join('images', 'people.jpg'))
        elif command == '2':
            folder_path = input('Enter the folder path containing images: ')
            DetectFacesInFolder(folder_path)

    except Exception as ex:
        print("Error:", ex)


def DetectFaces(image_file):
    """ Detect faces in a single image """
    print('Detecting faces in', image_file)

    # Ensure the file exists
    if not os.path.exists(image_file):
        print(f"Error: File {image_file} not found.")
        return

    # Specify facial features to be retrieved
    features = [FaceAttributeTypeDetection03.HEAD_POSE,
                FaceAttributeTypeDetection03.BLUR,
                FaceAttributeTypeDetection03.MASK]

    # Get faces
    with open(image_file, mode="rb") as image_data:
        detected_faces = face_client.detect(
            image_content=image_data.read(),
            detection_model=FaceDetectionModel.DETECTION03,
            recognition_model=FaceRecognitionModel.RECOGNITION04,
            return_face_id=False,
            return_face_attributes=features,
        )

    if len(detected_faces) > 0:
        print(len(detected_faces), 'faces detected.')
        SaveAndAnnotateFaces(image_file, detected_faces)
    else:
        print("No faces detected.")


def DetectFacesInFolder(folder_path):
    """ Detect faces in all images inside a folder """
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found.")
        return

    # Get all image files in the folder (JPG, PNG, JPEG)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]

    if not image_files:
        print("No image files found in the folder.")
        return

    print(f"Found {len(image_files)} images. Processing...")

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"\nProcessing: {image_file}")
        DetectFaces(image_path)


def SaveAndAnnotateFaces(image_file, detected_faces):
    """ Save the detected faces with annotations """
    fig = plt.figure(figsize=(8, 6))
    plt.axis('off')
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)
    color = 'lightgreen'
    face_count = 0

    for face in detected_faces:
        face_count += 1
        print(f'\nFace number {face_count}')
        print(f' - Head Pose (Yaw): {face.face_attributes.head_pose.yaw}')
        print(f' - Head Pose (Pitch): {face.face_attributes.head_pose.pitch}')
        print(f' - Head Pose (Roll): {face.face_attributes.head_pose.roll}')
        print(f' - Blur: {face.face_attributes.blur.blur_level}')
        print(f' - Mask: {face.face_attributes.mask.type}')

        # Draw bounding box
        r = face.face_rectangle
        bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
        draw.rectangle(bounding_box, outline=color, width=5)
        annotation = f'Face {face_count}'
        plt.annotate(annotation, (r.left, r.top), backgroundcolor=color)

    plt.imshow(image)

    # Save annotated image
    outputfile = image_file.replace('.jpg', '_detected.jpg')
    fig.savefig(outputfile)

    print(f'\nResults saved in {outputfile}')


if __name__ == "__main__":
    main()
