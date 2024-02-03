from PIL import Image
import pytesseract
import cv2
import numpy as np

# Replace 'path/to/image.jpeg' with the actual path to your JPEG image
image_path = '1.jpeg'

# Read the image using PIL
image = Image.open(image_path)
image_np = np.array(image)

# print image size
print(image.size)

# Preprocess the image: convert to grayscale and apply thresholding
# gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
# _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Convert the image to RGB if it's not already (necessary for pytesseract)
if image_np.ndim == 2:  # Grayscale image
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
else:
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)


# Use pytesseract to detect text and its bounding boxes with adjusted PSM
detections = pytesseract.image_to_data(image_rgb, lang='por+eng', config='--psm 11', output_type=pytesseract.Output.DICT)

# Initialize a list to store extracted text
extracted_texts = []

# Loop through detections and blur detected text areas
for i in range(len(detections['text'])):
    if int(detections['conf'][i]) > 60:  # Confidence threshold to filter weak detections
        x, y, w, h = detections['left'][i], detections['top'][i], detections['width'][i], detections['height'][i]
        extracted_texts.append(detections['text'][i])

        # Blur the detected text region
        roi = image_np[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
        image_np[y:y+h, x:x+w] = blurred_roi

        # Draw a red rectangle around the text
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red color in BGR and 2px thickness

# Convert the numpy array back to PIL image and save
blurred_image = Image.fromarray(image_np)

# Save the image with high quality
blurred_image.save('blurred_image.jpeg', 'JPEG', quality=100)


# Function to check if a word should start a new phrase
def starts_new_phrase(word):
    return word[0].isupper() or word[0].isdigit()


# Initialize an empty list to hold the phrases
phrases = []

# Initialize an empty string to hold the current phrase
current_phrase = ''

# Iterate through the list of words
for word in extracted_texts:
    # If the word starts a new phrase and the current phrase is not empty,
    # add the current phrase to the list of phrases and start a new one
    if starts_new_phrase(word) and current_phrase:
        phrases.append(current_phrase.strip())
        current_phrase = word + ' '
    else:
        # Otherwise, add the word to the current phrase
        current_phrase += word + ' '


# Add the last phrase to the list if it's not empty
if current_phrase:
    phrases.append(current_phrase.strip())


# Print the list of phrases
for phrase in phrases:
    print(phrase)
