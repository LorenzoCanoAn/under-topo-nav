import numpy as np
from functions import min_distance
from params import *
import rospy


def get_closest_angle(i_angle, list_of_angles):
    md = 2* np.math.pi
    closest_angle = None
    n_chosen = None
    for n, angle in enumerate(list_of_angles):
        d = min_distance(angle, i_angle)
        if d < md:
            n_chosen = n
            md = d
            closest_angle = angle
    return closest_angle, md, n_chosen


class Gallery:
    def __init__(self, tracker,  angle, tracking=True, name=None, max_confidence=MAX_CONFIDENCE):
        self.max_confidence = max_confidence
        self.parent_tracker = tracker
        self.angle = angle
        self.confidence = 1
        self.max_confidence_achieved = False
        self.angular_threshold = 20  # deg

    def update_with_angles(self, input_angles: np.ndarray):

        new_angle, ang_dist, idx_of_new_angle = get_closest_angle(
            self.angle, input_angles)

        if ang_dist > self.angular_threshold * np.math.pi / 180:
            self.confidence = max(self.confidence-1, 0)
            if self.max_confidence_achieved and self.confidence==0: 
                self.parent_tracker.start_transition(message="of angles:" + str(input_angles)+ "the closest is at "+  str(ang_dist/ np.math.pi *180))
            return input_angles
        else:
            self.confidence = min(self.confidence+1, self.max_confidence)
            self.angle = new_angle
            if self.confidence == self.max_confidence:
                if not self.max_confidence_achieved:
                    self.parent_tracker.start_transition(message="Max confidence in new gallery")
                self.max_confidence_achieved = True
            return np.delete(input_angles, idx_of_new_angle)

    def get_id(self):
        return self.parent_tracker.galleries.index(self)


class GalleryTracker:
    def __init__(self):
        self.n_objective = 0

        self.followed_gallery_idx = None
        self.back_gallery_idx = None

        self.in_transition = True
        self.galleries = []
        self.in_transition_counter = 0
        self.transition_message = "Void"
        self.starting_up = True

    def get_closest_gallery_to_angle(self, obj_angle):
        md = np.math.pi
        for n, gallery in enumerate(self.galleries):
            d = min_distance(obj_angle, gallery.angle)
            if d < md:
                md = d
                back_gallery = gallery
        return back_gallery

    def start_transition(self, message = ""):
        self.in_transition = True
        self.transition_message = message

    def set_instructions(self, instructions):
        self.topological_instructions = instructions

    def new_angles(self, angles):
        for gallery in self.galleries:
            angles = gallery.update_with_angles(angles)

        for i in range(0, self.galleries.__len__()).__reversed__():
            if self.galleries[i].confidence < 1:
                self.galleries.pop(i)
                self.update_key_galleries_idx(i)

        for angle in angles:
            self.galleries.append(Gallery(self, angle))

        if self.starting_up:
            self.starting_up = False
            md = np.math.pi
            for n, gallery in enumerate(self.galleries):
                d = min_distance(gallery.angle, np.math.pi)
                if d < md:
                    md = d
                    self.back_gallery_idx = n

    def get_angle(self):
        if self.in_transition:
            if self.n_objective < self.topological_instructions.__len__():
                if self.in_transition_counter == 0:
                    print("Transition: {}".format(self.transition_message))
                    pass
                if self.in_transition_counter < TRANSITION_STEPS:
                    self.in_transition_counter += 1
                    if self.followed_gallery_idx == None:
                        if self.back_gallery_idx != None:
                            return (self.galleries[self.back_gallery_idx].angle + np.math.pi) % (np.math.pi*2)
                        else:
                            return None
                    else:
                        return self.galleries[self.followed_gallery_idx].angle

                else:
                    self.select_gallery_to_follow()
                    self.in_transition = False
                    self.in_transition_counter = 0
                    return self.galleries[self.followed_gallery_idx].angle
            else:
                return None

        else:
            if self.followed_gallery_idx == None:
                return None
            else:
                return self.galleries[self.followed_gallery_idx].angle


    def select_gallery_to_follow(self):
        if self.back_gallery_idx != None or self.back_gallery_idx > self.galleries.__len__():
            back_gallery = self.galleries[self.back_gallery_idx]
        else:
            if self.followed_gallery_idx != None:
                obj_angle = (
                    self.galleries[self.followed_gallery_idx].angle + np.math.pi) % (np.math.pi*2)
                back_gallery = self.get_closest_gallery_to_angle(obj_angle)

        def key(element):
            return element.angle

        self.galleries.sort(key=key)

        self.back_gallery_idx = back_gallery.get_id()

        ti = self.topological_instructions[self.n_objective]
        print("{} of {} is {}".format(self.n_objective, self.topological_instructions, ti))

        self.followed_gallery_idx = self.back_gallery_idx + ti

        if self.followed_gallery_idx < 0:
            self.followed_gallery_idx = self.galleries.__len__() + self.followed_gallery_idx
        if self.followed_gallery_idx >= self.galleries.__len__():
            self.followed_gallery_idx =  self.followed_gallery_idx % self.galleries.__len__()

        self.n_objective += 1

    def update_key_galleries_idx(self, i):
        if i == self.followed_gallery_idx:
            self.followed_gallery_idx = None
        elif i < self.followed_gallery_idx:
            self.followed_gallery_idx -= 1

        if i == self.back_gallery_idx:
            obj_angle = (
                self.galleries[self.followed_gallery_idx].angle + np.math.pi) % (np.math.pi*2)
            self.back_gallery_idx = self.get_closest_gallery_to_angle(
                obj_angle).get_id()
        else:
            if i < self.back_gallery_idx:
                self.back_gallery_idx -= 1

    def print_confidences(self):
        print("front:{}\tback:{}".format(
            self.following.confidence, self.back.confidence), end="\t")
        for n, gallery in enumerate(self.galleries):
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
