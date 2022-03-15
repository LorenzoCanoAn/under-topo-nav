import numpy as np
from functions import min_distance
from params import *


class Gallery:
    def __init__(self, tracker,  angle, tracking=True, name=None, max_confidence=MAX_CONFIDENCE):
        self.max_confidence = max_confidence
        self.tracker = tracker
        self.name = name
        self.angle = angle
        self.confidence = 1
        self.tracking = tracking
        self.following_angle = 20  # deg

    def __del__(self):
        print("Deleted gallery at angle {:.2f}".format(self.angle))

    def update_with_angles(self, input_angles: np.ndarray):
        md = np.math.pi
        for n, angle in enumerate(input_angles):
            d = min_distance(angle, self.angle)
            if d < md:
                n_chosen = n
                md = d
                new_angle = angle

        if self.tracking:
            if md > self.following_angle * np.math.pi / 180:  # IF THE ANGLE THAT WAS BEING FOLLOWED DISSAPEARS
                self.confidence = max(self.confidence-1, 0)
                return input_angles
            else:
                if self.confidence == self.max_confidence - 1:
                    self.tracker.in_transition = True
                self.confidence = min(self.confidence+1, self.max_confidence)
                self.angle = new_angle
                return np.delete(input_angles, n_chosen)
        else:
            if md < 30 * np.math.pi / 180:
                self.tracking = True
                self.angle = new_angle
                self.confidence = 1
                return np.delete(input_angles, n_chosen)
            else:
                return input_angles

    def reset(self, new_angle, new_confidence=0):
        self.new_angle = new_angle
        self.confidence = new_confidence
        self.tracking = False


class GalleryTracker:
    def __init__(self):
        self.topografic_objectives = [1, 2, 1, 0]
        self.n_objective = 0
        self.angles = []
        self.in_transition = True
        self.following = Gallery(self, 0, tracking=False, name="following")
        self.following.following_angle = 10
        self.back = Gallery(self, np.math.pi, tracking=False, name="back")
        self.secondary_galleries = []
        self.force_counter = 0

    def new_angles(self, angles):
        angles = np.array(angles)

        # Check the change of the back
        angles = self.back.update_with_angles(angles)

        # Check the change in the gallery being followed
        angles = self.following.update_with_angles(angles)

        # Check other angles
        self.handle_secondary_galleries(angles)

    def handle_secondary_galleries(self, angles):
        
        for gallery in self.secondary_galleries:
            angles = gallery.update_with_angles(angles)

        for i in range(0, self.secondary_galleries.__len__()).__reversed__():
            print("hola")
            if self.secondary_galleries[i].confidence < 1:
                self.secondary_galleries.pop(i)
                self.in_transition = True

        for angle in angles:
            self.secondary_galleries.append(Gallery(self, angle))

    def get_angle(self):
        if self.in_transition:
            pass
        else:
            pass

        if self.following.confidence >= MAX_CONFIDENCE - 1:
            return self.following.angle
        else:
            if self.force_counter < 20:
                if self.force_counter % 5 == 0:
                    print("Forcing forward")
                self.force_counter += 1
                return (self.back.angle + np.math.pi) % (np.math.pi*2)
            else:
                return None

    def print_confidences(self):
        print("front:{}\tback:{}".format(
            self.following.confidence, self.back.confidence), end="\t")
        for n, gallery in enumerate(self.secondary_galleries):
            print("{}:{}".format(n, gallery.confidence), end="\t")
        print("")


# LOGICA A IMPLEMENTAR:
#   - Cuando una galería llege a una confidence de 0 o a una confidence máxima por primera vez
#       - Se entra en un periodo de transición en el que unicamente se avanza siguiendo lo contrario a el following. 
#       - La duración del periodo de transición debería ser suficiente para que todas las galerías que han dejado de verse
            # llegen a una confidence de 0 y desaparezcan y que todas las galerías que han comenzado a verse lleguen a una 
            # confidence de 5 y pasen a ser elegilbes
#   - Una vez se ha terminado el periodo de transicíon, se toma la decisión de que galería seguir. 
#   - Tras elegir que galería se va a seguir, también hay que elegir que galería será la trasera.

# DUDAS:
# - Que hacer si desaparece la galería "trasera"? No se me ocurre en que escenario podría ocurrir pero habría que ternerlo en cuenta.