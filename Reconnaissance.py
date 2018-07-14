from skimage.measure import compare_ssim as ssim
import os
import cv2 as cv
import glob


object_detected = "None"

list_xml_files = glob.glob('haarstages/*.xml')
video_capture = cv.VideoCapture('video/stp.mp4')


def calcul_sim(image_origine, dataset_subdir, h, w):
    image_origine_resized = cv.resize(image_origine, (450, 450))
    images = glob.glob(dataset_subdir)
    nb_image = 0
    somme = 0.0
    # print("Here in function")
    for image in images:
        image_to_compare_with = cv.imread(image)
        image_to_compare_with_resized = cv.resize(image_to_compare_with, (450, 450))
        image_to_compare_with_grayscale = cv.cvtColor(image_to_compare_with_resized, cv.COLOR_BGR2GRAY)
        somme += ssim(image_origine_resized, image_to_compare_with_grayscale)
        nb_image += 1

    print("La moyenne est : ", + somme / nb_image)
    return somme / nb_image
    pass


while True:
    red, image = video_capture.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for filename in list_xml_files:
        # print(filename)
        sign_cascade = cv.CascadeClassifier(filename)
        signs = sign_cascade.detectMultiScale(
            gray,
            scaleFactor=1.4,
            minNeighbors=3
        )
        if len(signs) != 0:
            for (x, y, w, h) in signs:
                obj = gray[y:y + h, x:x + w]
                if filename == 'haarstages\speedLimit.xml':
                    images_files = next(os.walk('data/SpeedLimit'))[1]
                    pourcentage = 0.0
                    seuil = 0.3
                    for sub in images_files:
                        new_pourcentage = calcul_sim(obj, 'data/SpeedLimit/' + sub + '/*', h, w)
                        if new_pourcentage >= seuil:
                            if pourcentage < new_pourcentage:
                                pourcentage = new_pourcentage
                                object_detected = sub

                else:
                    if filename == 'haarstages\Stop.xml':
                        print("Function")
                        if calcul_sim(obj, 'data/Stop/*', 450, 450) >= 0.3:
                            object_detected = 'Stop'
                    # else:
                    #     others_path = next(os.walk('data/'))[1]
                    #     for other in others_path:
                    #         if other != 'SpeedLimit' and other != 'Stop':
                    #             path = 'data/{}/*'.format(other)
                    #             print('Path : ', path)
                    #             if calcul_sim(obj, path) >= 0.8:
                    #                 object_detected = other
                if object_detected != 'None':
                    cv.putText(image, object_detected, (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                object_detected = 'None'
            cv.waitKey(0)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    image = cv.resize(image, (960, 540))
    cv.imshow("Video", image)
video_capture.release()
cv.destroyAllWindows()
