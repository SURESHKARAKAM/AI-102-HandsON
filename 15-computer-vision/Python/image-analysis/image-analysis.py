from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests

# Import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def main():
    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
        ai_key = os.getenv('COG_SERVICE_KEY')

        # Get image
        image_file = 'images/street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        with open(image_file, "rb") as f:
            image_data = f.read()

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )
        
        # Analyze image
        AnalyzeImage(image_file, image_data, cv_client)

    except Exception as ex:
        print(f"Error: {ex}")


def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Get result with specified features to be retrieved
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ],
        )

        # Get image captions
        if result.caption is not None:
            print("\nCaption:")
            print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))

        # Get image dense captions
        if result.dense_captions is not None:
            print("\nDense Captions:")
            for caption in result.dense_captions:
                print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))

        # ✅ Get image tags
        if hasattr(result, "tags") and result.tags:
            print("\nTags:")
            for tag in result.tags:
                print(" Tag: '{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))
        else:
            print("\nNo tags detected.")

        # ✅ Get objects in the image
        if hasattr(result, "objects") and result.objects:
            print("\nObjects in Image:")
            for obj in result.objects:
                print(" Object: '{}' (confidence: {:.2f}%)".format(obj.name, obj.confidence * 100))

        # ✅ Get people in the image
        if hasattr(result, "people") and result.people:
            print("\nPeople in image:")

            # Prepare image for drawing
            image = Image.open(image_filename)
            fig = plt.figure(figsize=(image.width/100, image.height/100))
            plt.axis('off')
            draw = ImageDraw.Draw(image)
            color = 'cyan'

            for detected_people in result.people:
                # Draw object bounding box
                r = detected_people.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)

            # Save annotated image
            plt.imshow(image)
            plt.tight_layout(pad=0)
            outputfile = 'people.jpg'
            fig.savefig(outputfile)
            print('  Results saved in', outputfile)

    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")


if __name__ == "__main__":
    main()
