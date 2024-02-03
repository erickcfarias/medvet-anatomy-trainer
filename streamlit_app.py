from PIL import Image
import pytesseract
import cv2
import numpy as np
import re
import random
import os
import streamlit as st
from openai import OpenAI

client = OpenAI()

# Function to check if a word should start a new phrase
def starts_new_phrase(word):
    return word and (word[0].isupper() or word[0].isdigit())


def select_random_phrase(phrases, image_np):

        # Filter phrases with more than 4 letters
        filtered_phrases = {phrase: coords for phrase, coords in phrases.items() if len(phrase) > 4}
        
        if filtered_phrases:
            # Select a random phrase from the filtered phrases
            if "phrase" not in st.session_state:
                random_phrase = random.choice(list(filtered_phrases.keys()))

            else:
                random_phrase = st.session_state['phrase'][0]

            # Get the coordinates of the selected phrase
            x_start = filtered_phrases[random_phrase]['x_start']
            y_start = filtered_phrases[random_phrase]['y_start']
            x_end = filtered_phrases[random_phrase]['x_end']
            y_end = filtered_phrases[random_phrase]['y_end']

            cv2.rectangle(image_np, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2) 

            st.session_state['phrase'] = (random_phrase, (x_start, y_start, x_end, y_end))

            return random_phrase


# Function to draw bounding boxes around phrases and store them in a dictionary
def apply_blur(image_np, detections):
    num_detections = len(detections['text'])
    i = 0
    phrases = {}  # Dictionary to store phrases and their coordinates

    for i in range(i, num_detections):

        if int(detections['conf'][i]) > 10:  # Confidence threshold

            x, y, w, h = detections['left'][i], detections['top'][i], detections['width'][i], detections['height'][i]

            if h > 40 and w > 40:
                continue
            elif w > 200:
                continue

            roi = image_np[y:y+h, x:x+w]

            # Calculate the kernel size as a fraction of the ROI's dimensions
            kernel_width = max(3, int(roi.shape[1] * 0.8) | 1)  # Ensure the kernel size is odd and at least 3
            kernel_height = max(3, int(roi.shape[0] * 0.8) | 1)  # Ensure the kernel size is odd and at least 3

            # Apply Gaussian Blur with the dynamic kernel size
            blurred_roi = cv2.GaussianBlur(roi, (kernel_width, kernel_height), 0)

            # Replace the ROI with the blurred version in the original image
            image_np[y:y+h, x:x+w] = blurred_roi

            if starts_new_phrase(detections['text'][i]):
                # Start of a new phrase
                x_start, y_start = detections['left'][i], detections['top'][i]
                x_end, y_end = x_start + detections['width'][i], y_start + detections['height'][i]

                # Continue until the end of the phrase
                a = i + 1
                phrase = detections['text'][i]
                while a < num_detections and not starts_new_phrase(detections['text'][a]) and detections['text'][a] != "":

                    phrase = phrase + " " + detections['text'][a]

                    x = detections['left'][a]
                    y = detections['top'][a]
                    w = detections['width'][a]
                    h = detections['height'][a]

                    # Expand the bounding box to include this word
                    x_start = min(x_start, x)
                    y_start = min(y_start, y)
                    x_end = max(x_end, x + w)
                    y_end = max(y_end, y + h)

                    a += 1

                # Store the phrase and its coordinates in the dictionary
                phrases[phrase] = {'x_start': x_start, 'y_start': y_start, 'x_end': x_end, 'y_end': y_end}

            else:
                i += 1
        else:
            i += 1

    return phrases


def check_answer(text, phrase):
    if 'tentativas' not in st.session_state:
        st.session_state['tentativas'] = 0
        st.session_state['acertos'] = 0

    st.session_state['tentativas'] += 1

    if text.lower() == phrase.lower():
        st.success('Parabéns! Você acertou!')
    else:
        completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "Você é um professor de anatomia animal, e precisa corrigir um exercício de uma de suas alunas. O exercício consiste em identificar uma parte do corpo de um animal, e a aluna respondeu o seguinte:"},
                    {"role": "user", "content": text.lower()},
                    {"role": "system", "content": f"A resposta correta é a seguinte: {phrase.lower()}."},
                    {"role": "system", "content": f"Considere os seguintes critérios: - SIM: a resposta da aluna é igual à resposta correta.  - PARCIAL: a resposta da aluna é parcialmente correta, mas não completamente. - NÃO: a resposta da aluna é diferente da resposta correta. Indique se a aluna respondeu corretamente, escrevendo exatamente SIM, NÃO ou PARCIAL"},

                ]
        )

        correction = completion.choices[0].message.content.lower()
        if (correction == 'sim.') | (correction == 'sim'):
            st.success(f'Parabéns! Você acertou!')
            st.session_state['acertos'] += 1

        elif (correction == 'parcial.') | (correction == 'parcial'):
            st.warning("Você acertou parcialmente! Tente na próxima!")
        else:
            st.error('Você errou! Tente na próxima!')

if __name__ == '__main__':
    # # Get the list of JPEG files in the "data" folder
    # data_folder = 'data'
    # jpeg_files = [file for file in os.listdir(data_folder) if file.endswith('.jpeg')]

    # # Select a random JPEG file
    # image_path = os.path.join(data_folder, random.choice(jpeg_files))
    # # image_path = "data/4.jpeg"

    # # Read the image using PIL
    # image = Image.open(image_path)
    # image_np = np.array(image)  # Convert PIL Image to NumPy array to work with OpenCV

    # # Preprocess the image: convert to grayscale and apply thresholding
    # gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    # _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imwrite('gray_image.jpeg', hsv)

    # # Use pytesseract to detect text and its bounding boxes
    # detections = pytesseract.image_to_data(thresh_image, lang='por+eng', config='--psm 11', output_type=pytesseract.Output.DICT)

    # # Draw bounding boxes around phrases
    # phrases = apply_blur(image_np, detections)

    # select_random_phrase(phrases, image_np)

    # # Convert the numpy array back to a PIL Image
    # annotated_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    # # Save the image with high quality
    # annotated_image.save('annotated_image_phrases.jpeg', 'JPEG', quality=95)

    st.markdown('### Anatomy Trainer Tabajara')

    data_folder = 'data'
    jpeg_files = [file for file in os.listdir(data_folder) if file.endswith('.jpeg')]

    if 'image' not in st.session_state:
        # Select a random JPEG file
        image_path = os.path.join(data_folder, random.choice(jpeg_files))
        # image_path = "data/escapula-equino.jpeg"

        image = Image.open(image_path)

        st.session_state['image'] = image

    image_np = np.array(st.session_state['image']) # Convert PIL Image to NumPy array to work with OpenCV

    with st.spinner('Processing...'):
        # Preprocess the image: convert to grayscale and apply thresholding
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # st.image(thresh_image, caption='Thresh Image', width=400)

        # Use pytesseract to detect text and its bounding boxes
        detections = pytesseract.image_to_data(thresh_image, lang='por', config='--psm 11', output_type=pytesseract.Output.DICT)

        # Draw bounding boxes around phrases and apply blur
        phrases = apply_blur(image_np, detections)
        phrase = select_random_phrase(phrases, image_np)

        # Convert the numpy array back to a PIL Image for display
        annotated_image = Image.fromarray(image_np)

    text = st.text_input('Qual o nome da parte indicado em vermelho?')

    c1, c2, c3 = st.columns([1, 1, 1])
    button1 = c1.button('Ver Resposta')
    button2 = c2.button('Próxima Imagem')
    button3 = c3.button('Quero saber mais')

    if button1:

        check_answer(text, phrase)

        image_np = np.array(st.session_state['image'])

        # Draw a red bounding box around the selected phrase
        phrase = st.session_state['phrase'][0]
        x_start = st.session_state['phrase'][1][0]
        y_start = st.session_state['phrase'][1][1]
        x_end = st.session_state['phrase'][1][2]
        y_end = st.session_state['phrase'][1][3]

        cv2.rectangle(image_np, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2) 
        annotated_image = Image.fromarray(image_np)

        st.image(annotated_image, caption='Original Image', width=800)

    else:
        st.image(annotated_image, caption='Processed Image', width=800)

    if button2:
        del st.session_state['image']
        del st.session_state['phrase']
        st.rerun()

    if button3:
        completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "Você é um professor de anatomia animal, de uma renomada universidade Portuguesa. Uma de suas alunas encontrou no livro texto a seguinte parte de um animal:"},
                    {"role": "user", "content": phrase.lower()},
                    {"role": "system", "content": "Explique brevemente o que é isso e qual a sua função no corpo do animal."}

                ]
            )

        st.write(completion.choices[0].message.content)