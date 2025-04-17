# Importing the libraries
import face_recognition
import cv2
import os

# Image preprocessing function
def img_resize(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read the image at path: {path}")
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(img, (width, height))

# Lists to store the face encodings and names
train_enc = []
train_names = []

# Training the model
training_images = 'train'
for file in os.listdir(training_images):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(training_images, file)
        img = img_resize(path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            train_enc.append(encodings[0])
            train_names.append(file.split('.')[0])
        else:
            print(f"No face found in training image: {file}")

# Testing the model
testing_images = 'test'
for file in os.listdir(testing_images):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(testing_images, file)
        img = img_resize(path)
        encodings = face_recognition.face_encodings(img)
        if not encodings:
            print(f"No face found in test image: {file}")
            continue

        img_enc = encodings[0]
        outputs = face_recognition.compare_faces(train_enc, img_enc)

        # Displaying the results
        face_locations = face_recognition.face_locations(img)
        for i in range(len(outputs)):
            if outputs[i]:
                name = train_names[i]
                if face_locations:
                    (top, right, bottom, left) = face_locations[0]
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(img, name, (left + 2, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                else:
                    print("No face location found for matched image.")

        # Show the image with result
        cv2.imshow(f"Result for {file}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
